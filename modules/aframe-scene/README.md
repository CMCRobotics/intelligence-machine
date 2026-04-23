# Intelligence Machine - A-Frame Scene

This module provides the 3D lab environment and terminal interfaces for the Intelligence Machine workshop.

## Deployment Configuration

The application can be configured via environment variables for server deployment and custom domains.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | The port the Node.js server listens on. | `3000` |
| `MQTT_BROKER_URL` | The MQTT broker URL for the backend server to connect to (e.g., `ws://mqtt-broker:9001`). | `ws://localhost:9001` |
| `MQTT_BROKER_URL_CLIENT` | The MQTT broker URL for the frontend (browser) to connect to. Should be a publicly reachable URL. | `ws://<hostname>:9001` |
| `API_URL` | The base URL for the model upload API, reachable by the browser. | `http://<hostname>:<PORT>` |

### Runtime Configuration

The application serves a dynamic `/config.js` file that injects these variables into the frontend at runtime. This allows for immutable Docker images that can be configured per environment (e.g., in Kubernetes or OpenShift using ConfigMaps or environment variables).

### Local Development

For local development, you can create a `.env` file based on `.env.example`. If no variables are set, the application defaults to `localhost` settings.
