# dixiyao-controlNetFL

For all the codes, remember to set the condition and proper path to save models in the file.
## Performance
### Original ControlNet
```
python train.py
```
### With trained autoencoder
```
python train_clip_as_condition.py
```
### And hide the prompts
```
python train_clip_hide_prompt.py
```

## Model Inversion attack
### Original model cut after mix
#### Step 1:
Train the estimated client model weights. 
```
python train_out_client.py
```
#### Step2:
Used the trained estimated client model in step 1 to train an inversion network.
```
python reconstruct.py
```
#### Step3:
Verify trained inversion network on other private dataset.
```
python reconstruct_valid.py
```

### With trained autoencoder
Because the autoencoder weights are pretrained, the server knows the client model weights
```
python reconstruct_after_encoder_clip.py
```
