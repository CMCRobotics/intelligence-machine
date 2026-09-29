import { LitElement, html, css } from 'lit';
import { createMqttHomieObserver } from '@cmcrobotics/homie-lit';
import { merge } from 'rxjs';

class TeachableMachineUploadView extends LitElement {
  static get properties() {
    return {
      name: { type: String },
      deviceId: { type: String },
      teamId: { type: String },
      modelName: { type: String },
      modelType: { type: String },
      uploadStatus: { type: String },
    };
  }

  constructor() {
    super();
    this.name = "Model Upload";
    this.uploadStatus = '';
    this.modelName = '';
    this.modelType = '';
  }

  connectedCallback() {
    super.connectedCallback();
    this.deviceId = localStorage.getItem('deviceId');
    this.teamId = localStorage.getItem('teamId');
    if (this.deviceId) {
      this.connect();
    }
  }

  connect() {
    if (!this.homieObserver) {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const mqttUrl = (window.APP_CONFIG && window.APP_CONFIG.MQTT_BROKER_URL) || `${scheme}://${window.location.hostname}:9001`;
      this.homieObserver = createMqttHomieObserver(mqttUrl);
      const nameTopic = `terminal-${this.deviceId}/model-upload/name`;
      const typeTopic = `terminal-${this.deviceId}/model-upload/type`;

      merge(
            this.homieObserver.created$,
            this.homieObserver.updated$
        ).subscribe(event => {
        if (event.type === 'property') {
          if (event.property.id === 'name' && event.property.value !== undefined) {
            this.modelName = event.property.value;
          } else if (event.property.id === 'type' && event.property.value !== undefined) {
            this.modelType = event.property.value;
          }
          this.requestUpdate();
        }
      });

      this.homieObserver.subscribe(nameTopic);
      this.homieObserver.subscribe(typeTopic);
    }
  }

  async _handleSubmit(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    
    this.uploadStatus = 'Uploading...';

    try {
      const apiUrl = (window.APP_CONFIG && window.APP_CONFIG.API_URL) || `http://${window.location.hostname}:3000`;
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        this.uploadStatus = result.message;
      } else {
        this.uploadStatus = `Error: ${response.statusText}`;
      }
    } catch (error) {
      this.uploadStatus = `Error: ${error.message}`;
    }
  }

  render() {
    const teamName = localStorage.getItem('teamName') || '';
    return html`
      <div class="view">
        <div class="header">
          <div class="team-badge" style="background-color: ${teamName.toLowerCase()}; color: ${this._getTextColor(teamName)}">
            Team ${teamName}
          </div>
          <h2>${this.name}</h2>
        </div>

        <form @submit="${this._handleSubmit}">
          <input type="hidden" name="team-id" .value="${this.teamId}">
          <input type="hidden" name="terminalId" .value="${this.deviceId}">
          
          <div class="form-row">
            <div class="form-field">
              <label for="name">Model Name</label>
              <input type="text" id="name" name="name" .value="${this.modelName}" readonly required>
            </div>

            <div class="form-field">
              <label for="name">Model Type</label>
              <input type="text" id="type" name="modelType" .value="${this.modelType}" readonly required>
            </div>
          </div>
          
          <div class="form-field file-field">
            <label for="model">Model File (.zip)</label>
            <input type="file" id="model" name="model" accept=".zip" required>
          </div>
          
          <button type="submit" class="${this.uploadStatus === 'Uploading...' ? 'loading' : ''}">
            ${this.uploadStatus === 'Uploading...' ? 'Uploading...' : 'Upload Model'}
          </button>
        </form>
        ${this.uploadStatus ? html`<p class="status-msg">${this.uploadStatus}</p>` : ''}
      </div>
    `;
  }

  _getTextColor(teamName) {
    if (!teamName) return 'white';
    const darkColors = ['blue', 'red', 'purple', 'green'];
    return darkColors.includes(teamName.toLowerCase()) ? 'white' : 'black';
  }

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
    }

    .view {
      padding: 30px;
      text-align: center;
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
      color: white;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 30px;
    }

    .team-badge {
      padding: 4px 15px;
      border-radius: 15px;
      font-weight: bold;
      font-size: 1rem;
      box-shadow: 0 2px 5px rgba(0,0,0,0.3);
      text-transform: uppercase;
    }

    h2 {
      margin: 0;
      font-size: 1.5rem;
      text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }

    form {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 20px;
      max-width: 500px;
      margin: 0 auto;
      width: 100%;
    }

    .form-row {
      display: flex;
      gap: 15px;
    }

    .form-field {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 5px;
    }

    .file-field {
      background: rgba(255, 255, 255, 0.05);
      padding: 15px;
      border-radius: 8px;
      border: 1px dashed rgba(255, 255, 255, 0.2);
    }

    label {
      font-size: 0.9rem;
      font-weight: bold;
      color: #ccc;
    }

    input[type="text"] {
      width: 100%;
      padding: 10px;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 4px;
      color: #fff;
      box-sizing: border-box;
    }

    input[type="file"] {
      width: 100%;
      color: #ccc;
    }

    button {
      padding: 12px;
      border: none;
      background-color: #2196F3;
      color: white;
      border-radius: 4px;
      cursor: pointer;
      font-size: 1.1rem;
      font-weight: bold;
      transition: background 0.3s;
      box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    button:hover {
      background-color: #1976D2;
    }

    button.loading {
      background-color: #666;
      cursor: not-allowed;
    }

    .status-msg {
      margin-top: 20px;
      font-style: italic;
      color: #4caf50;
    }
  `;
}

customElements.define('teachable-machine-upload-view', TeachableMachineUploadView);

export { TeachableMachineUploadView };
