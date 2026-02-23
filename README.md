# MIT 6.S184: Generative AI with Stochastic Differential Equations - Labs

This repository contains my solutions for the three labs from **MIT 6.S184: Generative AI with Stochastic Differential Equations**.

## 📚 Course Information

MIT 6.S184 focuses on generative AI techniques using stochastic differential equations, covering topics such as:

- Numerical methods for ODEs and SDEs
- Flow matching and diffusion models
- Conditional generation and classifier-free guidance
- Modern architectures like U-Net for image generation



## 🔗 Course Resources

- **Course Website**: [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/)
- **Course Videos**: Available on the course website.
- **Course Textbook**: [*An Introduction to Flow Matching and Diffusion Models*](https://arxiv.org/abs/2506.02070)
- **Course Assignments**: Three labs (details available on the course website).

---

## 🛠️ Bug Fixes & Troubleshooting

### Lab 2: Visualization `ValueError` Bug (Fixed)

**Issue:** The original visualization blocks in `lab_two.ipynb` had a bug where they passed PyTorch tensors directly into NumPy-based plotting functions, which triggered a `ValueError: range argument must have one entry per dimension`.

```python
# Original buggy code (Cell In[3]):
samples = sampleable.sample(num_samples).detach().cpu() # Bug: Returns a PyTorch Tensor, not a NumPy array
```
Since samples remains a PyTorch Tensor after calling .cpu(), NumPy's histogram2d misinterprets its shape and dimensions, causing the function to crash.

**Fix:** Changed to use .numpy() to explicitly convert the tensor to a NumPy array before plotting:

```python
# Fixed code:
samples = sampleable.sample(num_samples).detach().cpu().numpy()
```
**Issue:** The same missing conversion caused another crash later in the notebook during the conditional path sampling visualization.

```Python
# Original buggy code (Cell In[32]):
hist2d_samples(samples=xts.cpu(), ax=axes[0, idx], bins=300, scale=scale, percentile=percentile, alpha=1.0) # Bug: xts.cpu() is still a Tensor
```
**Fix:** Changed to append .numpy() to correctly format the input for the histogram function:

```Python
# Fixed code:
hist2d_samples(samples=xts.cpu().numpy(), ax=axes[0, idx], bins=300, scale=scale, percentile=percentile, alpha=1.0)
```

**Note:** The same fix should be applied to the other calls to hist2d_samples throughout the notebook.