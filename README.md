# VAE 艺术风格迁移与生成（PyTorch 手动实现）

该仓库提供一个**不依赖高级 VAE 封装库**的参考实现，包含：
- 手写 Encoder / Decoder / 重参数化。
- 重建损失（MSE）+ KL 散度损失。
- 50 轮训练 + 早停策略。
- 写实→印象派、印象派→写实、潜在向量融合、随机采样生成。
- 指标输出（MSE）与可选 FID（依赖 `torchmetrics`）。

## 1. 数据组织（ImageFolder 格式）
请分别准备两个根目录，每个目录下至少有一个子文件夹放图像：

```text
data/
  coco/
    all/
      xxx.jpg
  wikiart/
    all/
      yyy.jpg
```

> 代码会统一将图像 resize+center crop 到 256×256，像素映射到 [0,1]。

## 2. 训练命令

```bash
python vae_style_transfer.py \
  --real-data data/coco \
  --imp-data data/wikiart \
  --output-dir outputs \
  --epochs 50 \
  --batch-size 32 \
  --latent-dim 256 \
  --beta 1.0
```

## 3. 输出内容
`outputs/` 下会生成：
- `best_vae.pt`：最佳验证集权重。
- `metrics.json`：训练历史、测试 MSE、可选 FID。
- `samples/real_to_imp.png`：写实→印象派示例。
- `samples/imp_to_real.png`：印象派→写实示例。
- `samples/latent_mix_alpha_*.png`：不同融合比例的过渡风格。
- `samples/random_generation.png`：随机采样生成图。

## 4. 说明
- 数据划分采用 8:1:1（按两个域分别切分再合并）。
- 为保证可复现，默认 seed=42。
- FID 仅在安装 `torchmetrics` 且提供足量真实/生成图像时更有意义。
