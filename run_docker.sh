sudo docker container stop container2
sudo docker container rm container2
sudo docker container run -it  -d --name container2 -p 8081:8081 -v $(pwd)/artifacts:/workspace/artifacts recommender-engine
