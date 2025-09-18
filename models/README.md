# Pretrained Models

The pretrained models are not stored on GitHub due to file size limitations.  
You can download them directly from Zenodo:  

👉 [Download models on Zenodo](https://doi.org/10.5281/zenodo.17137746)  

Once downloaded, extract the `models.zip` archive into the root of this repository.

---

## Model Naming Convention

Each model filename is composed of **three parts** separated by underscores:  

1. **Model type**  
   - `murso` = Mobile-URSONet  
   - `mursop` = Mobile-URSONet+  

2. **Precision**  
   - `fp32` = full precision (32-bit float)  
   - `int8` = 8-bit quantization  
   - `mpq` = mixed precision quantization  
   - `qat` = quantization-aware training  

3. **Dataset**  
   - `speed` = trained on the SPEED dataset  
   - `dspeed` = trained on the D-SPEED dataset  

**Example**:  
`murso_fp32_speed` → Mobile-URSONet, full precision, trained on SPEED.  
