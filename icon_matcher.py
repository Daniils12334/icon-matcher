#!/usr/bin/env python3
import argparse
import os
import numpy as np # type: ignore
from PIL import Image # type: ignore
import cv2 # type: ignore
import re
import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
from torchvision import models # type: ignore
import torch.nn.functional as F # type: ignore
import json
from tqdm import tqdm # type: ignore
import csv
from skimage.metrics import structural_similarity as ssim # type: ignore
import imagehash # type: ignore
import matplotlib.pyplot as plt # type: ignore

class IconMatcher:
    def __init__(self, device='auto', model_type='resnet50', debug=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == 'auto' else torch.device(device)
        self.model = self._load_model(model_type)
        self.ref_features = {}
        self.debug = debug
        self.weight_feat = 0.65  # Increased feature weight
        self.weight_ssim = 0.25
        self.weight_hash = 0.10
        
        # Unified transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.debug = debug
        self.weight_feat = torch.nn.Parameter(torch.tensor(0.6))
        self.weight_ssim = torch.nn.Parameter(torch.tensor(0.3))
        self.weight_hash = torch.nn.Parameter(torch.tensor(0.1))
        if self.debug:
            os.makedirs("preprocessed_debug", exist_ok=True)

    def _load_model(self, model_type):
        if model_type == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Deeper feature extraction
            return torch.nn.Sequential(*list(model.children())[:-2],  # Layer 5 features
                    torch.nn.AdaptiveAvgPool2d((1, 1))).eval().to(self.device)

    def preprocess_image(self, img_path, image_size=224):
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # Preserve transparency
            img = background
        else:
            img = img.convert("RGB")
        gray = np.array(img.convert("L"))
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 5)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:  # Fallback for empty threshold
            contours, _ = cv2.findContours(255-bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = [c for c in contours if cv2.contourArea(c) > 50]
            if contours:  # Additional check after filtering
                x, y, w, h = cv2.boundingRect(np.concatenate(contours))
                img = img.crop((x, y, x+w, y+h))
        w, h = img.size
        max_side = max(w, h)
        pad_w = (max_side - w) // 2
        pad_h = (max_side - h) // 2
        new_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))
        new_img.paste(img, (pad_w, pad_h))

        # Force resize to target dimension
        new_img = new_img.resize((image_size, image_size), Image.BILINEAR)

        if self.debug:
            base_name = os.path.basename(img_path)
            debug_path = os.path.join("preprocessed_debug", base_name)
            new_img.save(debug_path)

        tensor = self.transform(new_img)
        return new_img, tensor 

    def extract_features(self, img_tensor):
        with torch.no_grad():
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            features = self.model(img_tensor.to(self.device))
            features = features.mean([2, 3])
            return F.normalize(features, p=2, dim=1).cpu().numpy()

    def load_reference_images(self, ref_folder):
        print(f"Loading reference images from {ref_folder}...")
        self.ref_features = {}
        
        for fname in tqdm(os.listdir(ref_folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(ref_folder, fname)
            base_img, _ = self.preprocess_image(path)
            
            # Enhanced augmentation
            for angle in [0, 90, 180, 270]:
                for flip in [False, True]:
                    img = base_img.rotate(angle, expand=True)
                    if flip:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # Feature extraction
                    tensor = self.transform(img)
                    features = self.extract_features(tensor).flatten()
                    
                    self.ref_features[f"{fname}_aug{angle}{'_flip' if flip else ''}"] = {
                        "features": features,
                        "image": img,
                        "phash": imagehash.phash(img)
                    }      

    def hybrid_score(self, test_feat, test_img, test_phash, ref_data):
        # Enhanced feature similarity with cosine distance
        feat_sim = (np.dot(test_feat, ref_data["features"]) + 1) / 2
        
        # Color SSIM calculation
        try:
            test_np = np.array(test_img)
            ref_np = np.array(ref_data["image"])
            ssim_score = ssim(test_np, ref_np, 
                            channel_axis=-1,
                            win_size=3,
                            data_range=255)
        except Exception as e:
            ssim_score = 0

        # Combined hash comparison
        hash_score = 0.7 * (1 - (test_phash - ref_data["phash"])/64) + \
                   0.3 * (1 - (imagehash.average_hash(test_img) - 
                              imagehash.average_hash(ref_data["image"]))/64)
        
        return (self.weight_feat * feat_sim +
                self.weight_ssim * ssim_score +
                self.weight_hash * hash_score)
    
    def find_matches(self, test_path, top_k=3):
        test_img, tensor = self.preprocess_image(test_path)
        test_feat = self.extract_features(tensor).flatten()
        test_phash = imagehash.phash(test_img)

        scores = []
        for ref_name, ref_data in self.ref_features.items():
            score = self.hybrid_score(test_feat, test_img, test_phash, ref_data)
            scores.append((ref_name, score, ref_data["image"]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k], test_img
    
    def save_gif(self, test_img, test_name, matches):
        frames = [test_img] + [ref_img for _, _, ref_img in matches]
        resized = [img.resize((256, 256)) for img in frames]

        vis_dir = "visualizations"
        os.makedirs(vis_dir, exist_ok=True)

        safe_name = re.sub(r'[\\/*?:"<>|]', "_", test_name)
        gif_path = os.path.join(vis_dir, f"{safe_name}.gif")
        resized[0].save(gif_path, save_all=True, append_images=resized[1:], duration=1000, loop=0)

    def match_all(self, test_folder, top_k=3, output_format='csv', save_gif=False):

        results = []

        for fname in tqdm(sorted(os.listdir(test_folder))):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            test_path = os.path.join(test_folder, fname)
            matches, test_img = self.find_matches(test_path, top_k)

            # Debug: Log top 10 matches for all or specific image (e.g., 63.png)
            if self.debug and (fname == "63.png"):
                print(f"\nDEBUG: Top matches for {fname}:")
                top_matches = sorted([(m[0], m[1]) for m in self.find_matches(test_path, top_k=10)[0]], key=lambda x: x[1], reverse=True)
                for i, (ref_name, score) in enumerate(top_matches):
                    print(f"{i+1}. {ref_name}: {score:.4f}")


            for match in matches:
                results.append({
                    "test_image": fname,
                    "match": match[0].split("_rot")[0],  # Remove rotation suffix
                    "score": match[1]
                })
            self.visualize_matches(test_img, fname, matches)

            if save_gif:
                self.save_gif(test_img, fname, matches)

        return results

    def visualize_matches(self, test_img, test_name, matches):
        fig, axes = plt.subplots(1, len(matches)+1, figsize=(4*(len(matches)+1), 4))
        axes[0].imshow(test_img)
        axes[0].set_title(f"Test: {test_name}")
        axes[0].axis("off")
        for i, (ref_name, score, ref_img) in enumerate(matches):
            axes[i+1].imshow(ref_img)
            axes[i+1].set_title(f"{ref_name}\nScore: {score:.3f}")
            axes[i+1].axis("off")
        plt.tight_layout()
        vis_dir = "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, f"{test_name}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Enhanced Icon Matching System")
    parser.add_argument("--test_folder", required=True, help="Path to test images")
    parser.add_argument("--ref_folder", required=True, help="Path to reference images")
    parser.add_argument("--top_k", type=int, default=3, help="Number of matches to return")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--format", choices=['json', 'csv'], default='csv', help="Output format")
    parser.add_argument("--gif", action="store_true", help="Save animated GIFs of matches")
    parser.add_argument("--debug", action="store_true", help="Save preprocessed images and log detailed debug info")
    args = parser.parse_args()

    matcher = IconMatcher(debug=args.debug)
    matcher.load_reference_images(args.ref_folder)
    results = matcher.match_all(args.test_folder, args.top_k, args.format, save_gif=args.gif)

    if args.output:
        if args.format == 'json':
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['test_image', 'match', 'score'])
                writer.writeheader()
                for item in results:
                    writer.writerow(item)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
