# shawn_scripts — Gen3 Lite on the MnRI lab PC

## Where things live

| Lab PC (host)                                                   | Inside container                     |
|-----------------------------------------------------------------|--------------------------------------|
| `~/shawn/kinova-gen3-docker-ros/kinova_ws/`                     | `/root/kinova_ws/`                   |
| `~/shawn/kinova-gen3-docker-ros/kinova_ws/src/shawn_scripts/`   | `/root/kinova_ws/src/shawn_scripts/` |

Container name: `kinova-shawn`. Image: `kinova-gen3-dev:latest`.
Anything under `kinova_ws` survives even if the container is deleted.

## First time

On the lab PC (not in Docker):
```bash
mkdir -p ~/shawn
git clone https://github.com/umn-cse-mnri/kinova-gen3-docker-ros ~/shawn/kinova-gen3-docker-ros
mkdir -p ~/shawn/kinova-gen3-docker-ros/kinova_ws/src/shawn_scripts
# put these files in that shawn_scripts folder, then:
bash ~/shawn/kinova-gen3-docker-ros/kinova_ws/src/shawn_scripts/start_container.sh
```
Inside the container:
```bash
bash /root/kinova_ws/src/shawn_scripts/build_ws.sh
```

## Every session

Terminal 1 (lab PC):
```bash
bash ~/shawn/kinova-gen3-docker-ros/kinova_ws/src/shawn_scripts/start_container.sh
bash /root/kinova_ws/src/shawn_scripts/start_driver.sh
```
Terminal 2 (lab PC) — same start command opens a second shell in the running container:
```bash
bash ~/shawn/kinova-gen3-docker-ros/kinova_ws/src/shawn_scripts/start_container.sh
cd /root/kinova_ws/src/shawn_scripts
ROS_NAMESPACE=my_gen3_lite python3 one_move.py
```

## Safety
Gamepad **RB = E-stop**, **LB = clear fault**. Scripts default to 10% speed.
