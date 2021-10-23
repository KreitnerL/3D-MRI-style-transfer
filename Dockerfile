FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

COPY . /home/3D-MRI-style-transfer
RUN pip install -r /home/3D-MRI-style-transfer/requirements.txt

RUN echo "Successfully build image!"

WORKDIR /home/3D-MRI-style-transfer
CMD ["python", "/home/3D-MRI-style-transfer/test.py", "--name", "pix2pix_bayesian", "--gpu_ids", "-1", "--dataroot", "/var/dataset", "--bayesian", "--mean", "52.6", "--std", "89.12", "--dataset_mode", "mri", "--model", "pix2pix", "--netG", "obelisk", "--netD", "obelisk"]