from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
# suppress partial model loading warning
logging.set_verbosity_error()
import torch.nn as nn
import argparse
from util import *
import torchvision.transforms as T
from torch.nn import functional as F
import clip
import torchvision

model, preprocess = clip.load("ViT-B/32", device='cuda')
preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224),
                                                                           interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                                             torchvision.transforms.Normalize(
                                                 (0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711))])

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, opt, hf_key=None):
        super().__init__()
        self.opt = opt
        self.device = device
        self.sd_version = opt.sd_version
        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.model_key = model_key
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", revision="fp16",
                                                 torch_dtype=torch.float16).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                   torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print(f'[INFO] loaded stable diffusion!')

    def calculate_forward(self, x, cond, i, tj, timesteps):
        cond_batch = cond.repeat(x.shape[0], 1, 1)
        alpha_prod_t = self.scheduler.alphas_cumprod[tj]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[timesteps[i - 1]]
            if i > 0 else self.scheduler.final_alpha_cumprod
        )

        mu = alpha_prod_t ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        eps = self.unet(x, tj,
                        encoder_hidden_states=cond_batch).sample
        pred_x0 = (x - sigma_prev * eps) / mu_prev
        return mu * pred_x0 + sigma * eps

    def calculate_sample(self, x, cond, i, tj, timesteps):
        model_input = x
        cond_batch = cond.repeat(x.shape[0], 1, 1)

        alpha_prod_t = self.scheduler.alphas_cumprod[tj]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[timesteps[i + 1]]
            if i < len(timesteps) - 1
            else self.scheduler.final_alpha_cumprod
        )
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        eps = self.unet(model_input, tj, encoder_hidden_states=cond_batch).sample
        pred_x0 = (x - sigma * eps) / mu
        return mu_prev * pred_x0 + sigma_prev * eps


    @torch.autocast('cuda')
    def prime(self, x, eps, t, paths, prev_target=None, counter=0, prev_latent=None):
        n, c, h, w = x.shape
        alpha = 1.0
        eps_list = [eps] * 1000
        total_grad = 0.0
        weight = [10000] * 1000
        prev_score = None
        N = self.opt.step

        timesteps_re = reversed(self.scheduler.timesteps)
        timesteps = self.scheduler.timesteps
        cond = self.get_text_embeds('', '')[1].unsqueeze(0)
        pred_ori = self.vae.encode(2 * x - 1).latent_dist.mean.detach()

        if prev_latent is not None:
            pred_ori_new = pred_ori.view(n, -1).repeat(prev_latent.shape[0], 1)
            cosine_sim = torch.nn.functional.cosine_similarity(prev_latent, pred_ori_new, dim=-1)
            #get the maximum cosine similarity value
            max_cosine_sim = torch.max(cosine_sim)
            print('max cosine similarity: ', max_cosine_sim)
            if max_cosine_sim < 0.4:
                prev_latent = torch.cat([prev_latent, pred_ori.detach().view(n, -1)], dim=0)
                return x, paths, None, counter, prev_latent

        x_adv = x.detach() * 255. + torch.FloatTensor(*x.shape).uniform_(-eps, eps).to(torch.float16).cuda()

        counter += 1
        sample_num = 100
        paths_selected = random.sample(paths, n * sample_num)
        #remove selected paths from path list
        for path in paths_selected:
            paths.remove(path)

        images = []
        for path in paths_selected:
            image = Image.open(path).convert('RGB')
            image = image.resize((self.opt.W, self.opt.H), resample=Image.Resampling.LANCZOS)
            image = T.ToTensor()(image).to(torch.float16).cuda()
            images.append(image)
        target = torch.stack(images)

        with torch.no_grad():
            target_processed = preprocess(target)
            x_processed = preprocess(x)
            if prev_target is not None:
                prev_target_processed = preprocess(prev_target)

            target_features = model.encode_image(target_processed)
            target_features = target_features.view(n, sample_num, -1)
            x_features = model.encode_image(x_processed)
            x_features = x_features.repeat(1, sample_num).view(n, sample_num, -1)
            if prev_target is not None:
                prev_target_features = model.encode_image(prev_target_processed)
                prev_target_features = prev_target_features.repeat(1, sample_num).view(n, sample_num, -1)
            cosine_similarity = torch.nn.functional.cosine_similarity(target_features, x_features, dim=-1)
            if prev_target is not None:
                cosine_similarity += torch.nn.functional.cosine_similarity(target_features, prev_target_features, dim=-1)
            #select the image with the lowest cosine similarity as the new target for each image in batch
            target = target.view(n, -1, 3, h, w)
            new_target = target[:, torch.argmin(cosine_similarity, dim=1), :, :, :].squeeze(1)
            print('cosine similarity: ', cosine_similarity[:, torch.argmin(cosine_similarity, dim=1)])
        target = new_target

        pred_target = self.vae.encode(target * 2.0 - 1.0).latent_dist.mean.detach()
        with torch.no_grad():
            latent = pred_target * 0.18215
            intermediate_gt = [pred_target, latent]
            for i, tj in enumerate(timesteps_re):
                if i < t:
                    intermediate_gt.append(self.calculate_forward(intermediate_gt[-1], cond, i, tj, timesteps_re).detach())
            for i, tj in enumerate(timesteps):
                if i < t:
                    intermediate_gt.append(self.calculate_sample(intermediate_gt[-1], cond, i, tj, timesteps).detach())
            latents = 1 / 0.18215 * intermediate_gt[-1]
            intermediate_gt.append(self.vae.decode(latents).sample.detach())
            intermediate_gt.append(self.vae.decode(pred_target).sample.detach())

        with torch.enable_grad():
            for it in range(N):
                eps = eps_list[it]
                grad = 0.0
                x_adv_temp = x_adv.detach().clone()
                for eot in range(1):
                    x_adv_temp.requires_grad_()
                    pred_ = self.vae.encode(2. * (x_adv_temp / 255.) - 1.).latent_dist
                    pred = pred_.mean
                    intermediate = [pred, pred * 0.18215]

                    for i, tj in enumerate(timesteps_re):
                        if i < t:
                            intermediate.append(self.calculate_forward(intermediate[-1], cond, i, tj, timesteps_re))
                    for i, tj in enumerate(timesteps):
                        if i < t:
                            intermediate.append(self.calculate_sample(intermediate[-1], cond, i, tj, timesteps))

                    latents_pred = 1 / 0.18215 * intermediate[-1]
                    intermediate.append(self.vae.decode(latents_pred).sample)
                    intermediate.append(self.vae.decode(pred).sample)

                    loss = 0.0

                    for i in range(len(intermediate)):
                        if i != 1:
                            loss -= F.l1_loss(intermediate[i], intermediate_gt[i].detach()) * weight[i]

                    grad_temp = torch.autograd.grad(loss, [x_adv_temp])[0]
                    grad = grad + grad_temp * 10.0
                    x_adv_temp = x_adv.detach().clone()
                #break when all gradients are zero
                if torch.abs(grad).sum() == 0.0:
                    for i in range(len(weight)):
                        weight[i] *= 1.1
                #whether nan in gradient
                if torch.isnan(grad).sum() > 0:
                    x_adv = x.detach() * 255. + torch.FloatTensor(*x.shape).uniform_(-eps * 2.0, eps * 2.0).to(torch.float16).cuda()
                    weight = [10000] * 1000
                else:
                    x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
                    x_adv = torch.min(torch.max(x_adv, x * 255. - eps), x * 255. + eps)
                    x_adv = torch.clamp(x_adv, 0., 255.)
                    total_grad += grad.detach()

                if (it + 1) % 5 == 0:
                    if prev_latent is not None:
                        pred_ori_new = pred.detach().view(n, -1).repeat(prev_latent.shape[0], 1)
                        cosine_sim = torch.nn.functional.cosine_similarity(prev_latent, pred_ori_new, dim=-1)
                        # get the maximum cosine similarity value
                        max_cosine_sim = torch.max(cosine_sim)
                        if prev_score is None:
                            prev_score = max_cosine_sim.item()
                        else:
                            if (max_cosine_sim.item() > prev_score) or (max_cosine_sim.item() < 0.4):
                                print('max cosine similarity after attack %d iter: ' % (it + 1), max_cosine_sim.item())
                                break
                            else:
                                prev_score = max_cosine_sim.item()
                        print('max cosine similarity after attack %d iter: '%(it+1), max_cosine_sim.item())

        pred_ = self.vae.encode(2. * (x_adv / 255.) - 1.).latent_dist
        pred = pred_.mean

        print('latent diff: ', F.l1_loss(pred, pred_ori).mean().item())
        print('decode diff:', F.l1_loss((self.vae.decode(pred).sample.detach() / 2.0 + 0.5).clamp(0, 1) * 255.,
                                         x_adv).mean().item())

        rand_end = random.randint(0, 1000)
        import torchvision, imageio
        torchvision.utils.save_image(x_adv / 255., 'input_%d.png'%(int(rand_end)))
        torchvision.utils.save_image((self.vae.decode(pred).sample.detach() / 2.0 + 0.5).clamp(0, 1), 'output.png')

        #read x_adv from input.png and calculate the distance between input.png and x_adv
        x_adv_img = torchvision.io.read_image('input_%d.png'%(int(rand_end))).to(torch.float16).to(self.device)
        #print(x_adv_img.max())
        print('img diff:', F.l1_loss(x_adv_img, x_adv.squeeze(0)).mean().item())
        frame = torch.clamp(x_adv.squeeze(0).permute(1, 2, 0).cpu(), 0, 255).numpy().astype(np.uint8)
        imageio.mimsave('input_%d.mp4'%int(rand_end), [frame], fps=1, quality=10)

        from torchvision.io import read_video
        video, _, _ = read_video('input_%d.mp4'%int(rand_end), output_format="TCHW")
        image = T.ToPILImage()(video[0])
        image = image.resize((self.opt.W, self.opt.H),  resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(torch.float16).cuda()
        print('video img diff:', F.l1_loss(image * 255., x_adv.squeeze(0)).mean().item())

        pred_image = self.vae.encode(2. * image.unsqueeze(0) - 1.).latent_dist
        pred_image = pred_image.mean * 0.18215

        print('video latent diff:', F.l1_loss(pred_image, pred_ori).mean().item())
        print('video decode diff:', F.l1_loss((self.vae.decode(1 / 0.18215 * pred_image).sample.detach() / 2.0 + 0.5).clamp(0, 1) * 255.,
                                            image.unsqueeze(0) * 255.).mean().item())
        torchvision.utils.save_image((self.vae.decode(1 / 0.18215 * pred_image).sample.detach() / 2.0 + 0.5).clamp(0, 1), 'video_output_%d.png'%int(eps))

        if prev_latent is not None:
            pred_ori_new = 1 / 0.18215 * pred_image.detach().view(n, -1).repeat(prev_latent.shape[0], 1)
            cosine_sim = torch.nn.functional.cosine_similarity(prev_latent, pred_ori_new, dim=-1)
            #get the maximum cosine similarity value
            max_cosine_sim = torch.max(cosine_sim)
            print('max cosine similarity after attack: ', max_cosine_sim.item(), 'eps after attack: ', eps)

        if prev_latent is not None:
            #add pred_image.mean to prev_latent
            prev_latent = torch.cat([prev_latent, (1 / 0.18215 * pred_image.detach()).view(n, -1)], dim=0)
        else:
            prev_latent = (1 / 0.18215 * pred_image.detach()).view(n, -1)

        return x_adv.detach() / 255., paths, target, counter, prev_latent


    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def get_data(self, frames_path, n_frames, eps=None, ft=None):
        # load frames
        paths = [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
        if not os.path.exists(paths[0]):
            paths = [f"{frames_path}/%05d.jpg" % i for i in range(n_frames)]
        self.paths = paths
        frames = [Image.open(path).convert('RGB') for path in paths]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        if True:
            prev_target = None
            counter = 0
            prev_latent = None
            path = './val'
            paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPEG')]
            for i in range(len(frames)):
                (frames[i], paths, prev_target,
                     counter, prev_latent) = self.prime(frames[i].unsqueeze(0), eps,
                                                    ft, paths, prev_target, counter,
                                                                  prev_latent)
            import imageio
            video_frames = []
            for i in range(len(frames)):
                video_frames.append(torch.clamp(frames[i].detach().cpu().permute(1,2,0) * 255, 0, 255).numpy().astype(np.uint8))
            imageio.mimsave(self.opt.save_dir, video_frames, fps=30, quality=10)
        return

    def launch_protect(self,
                        ft):
        self.scheduler.set_timesteps(ft)
        eps = [8.]
        for e in eps:
            self.get_data(self.opt.data_path, self.opt.n_frames, e, ft)

def prep(opt):
    seed_everything(1)
    model = Preprocess(device, opt)
    model.launch_protect(ft=opt.ft)



if __name__ == "__main__":
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/woman-running.mp4') 
    parser.add_argument('--H', type=int, default=512, 
                        help='for non-square videos, we recommand using 672 x 384 or 384 x 672, aspect ratio 1.75')
    parser.add_argument('--W', type=int, default=512, 
                        help='for non-square videos, we recommand using 672 x 384 or 384 x 672, aspect ratio 1.75')
    parser.add_argument('--save_dir', type=str, default='latents')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1', 'ControlNet', 'depth'],
                        help="stable diffusion version")
    parser.add_argument('--n_frames', type=int, default=40)
    parser.add_argument('--ft', type=int, default=2)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--config_path', type=str, default='configs/config_attack_taylor.yaml')
    opt = parser.parse_args()
    import yaml
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)

    opt.data_path = config['raw_path']
    opt.sd_version = config['sd_version']
    opt.n_frames = config['n_frames']
    opt.save_dir = config['save_dir']
    opt.H = config['H']
    opt.W = config['W']

    opt.step = config['step']
    opt.ft = config['diffusion_steps']

    video_path = opt.data_path
    save_video_frames(video_path, Path(opt.save_dir).stem, img_size=(opt.W, opt.H))
    opt.data_path = os.path.join('data', Path(opt.save_dir).stem, Path(video_path).stem)
    opt.save_dir = config['raw_path'].replace('.mp4', f'_adv_{opt.sd_version}.mp4')

    print(opt.data_path, opt.save_dir)
    prep(opt)
