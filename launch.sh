sudo docker container rm exp_pas1
sudo docker build -t PAS .
sudo nvidia-docker run \
    --name exp_pas1 \
    --shm-size=50gb \
    -v /home/rst/dataset/vox1_musan:/data/vox1_musan \
    -v /home/rst/dataset/voxceleb1:/data/voxceleb1 \
    -t PAS:latest