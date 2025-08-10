# :brain: Intelligence Machine 


## :gear: How to develop

In order to obtain the latest dev version of dependencies, you may need a Github token.

### 🛠️ Creating a GitHub Personal Access Token for npm

* Go to GitHub.com and log in to your account.
* Click on your profile picture in the top-right corner and select "Settings".
* In the left sidebar, click on "Developer settings".
* Click on "Personal access tokens" and then "Tokens (classic)".
* Click "Generate new token" and then "Generate new token (classic)".
* Give your token a descriptive name.
    For scopes, select at minimum:
    ```
    read:packages
    ```
    If you also need to publish packages, include:
    ```
    write:packages
    delete:packages
    ```
* Click "Generate token" at the bottom of the page.
* Copy the token immediately - you won't be able to see it again!

You need to export your token as **NPM_AUTH_TOKEN**.


### How to reset your broker's persistent data (Mosquitto)

```bash
sudo service mosquitto stop && sudo rm /var/lib/mosquitto/mosquitto.db && sudo service mosquitto start
```

## 🔬 Technical Components

- WebXR for augmented reality experiences
- Custom AR utilities for object placement
- Linear animation system for smooth transitions
- Dynamic asset loading with fragment support

## 📱 Compatibility

- Works on WebXR-enabled browsers
- Supports both mobile and desktop devices
- Requires camera access for AR features

## 📄 License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.

---
