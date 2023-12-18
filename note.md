# Note for model
- SA: Model trained on Scenery6k with Adv
- SN: Model trained on Scenery6k without Adv
- WA: Model trained on Wikiart with Adv
- WN: Model trained on Wikiart without Adv 

# Train configuration
- Machine specs:
    - Ram: 32Gb
    - GPU: NVIDIA RTX 3090 24Gb
    - CPU: Threadripper 3960x
- Training:
    - Scenery: Split train, test, val -> 4k | 1k | 1k
    - Wikiart: Split train, test, val -> 56k | 12k | 12k

    SA: 200 epochs
    SN: 200 epochs
    WA: 152 epochs
    WN: 152 epochs

# Evaluation guide
- Map the file `log.csv` in each folder of `eval-html-{code}/0` with code is the model trained above.
- The image are stored in `eval-html-{code}/0/imgs`

 
# FID Score
- All train on its training dataset
- Scenary: eval with 1000 images
- Wikiart: eval with 500 images 
- The larger number images to evaluate, the better result but requires more ram -> not enough resources.
    - SA: 66.5201 
    - SN: 86.1417
    - WA: 141.9387
    - WN: 161.6858

