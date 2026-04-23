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
          if (event.property.id === 'name') {
            this.modelName = event.property.value;
          } else if (event.property.id === 'type') {
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
    return html`
      <div class="view">
        <h2>${this.name}</h2>
        <form @submit="${this._handleSubmit}">
          <input type="hidden" name="team-id" .value="${this.teamId}">
          <input type="hidden" name="terminalId" .value="${this.deviceId}">
          
          <div class="form-field">
            <label for="name">Model Name:</label>
            <input type="text" id="name" name="name" .value="${this.modelName}" readonly required>
          </div>

          <div class="form-field">
            <label for="name">Model Type:</label>
            <input type="text" id="type" name="modelType" .value="${this.modelType}" readonly required>
          </div>
          
          <div class="form-field">
            <label for="model">Model File (.zip):</label>
            <input type="file" id="model" name="model" accept=".zip" required>
          </div>
          
          <button type="submit">Upload</button>
        </form>
        ${this.uploadStatus ? html`<p>${this.uploadStatus}</p>` : ''}
      </div>
    `;
  }

  static styles = css`
    .view {
      padding: 20px;
      border: 1px dashed blue;
      background-color: #e3f2fd;
      text-align: center;
      width: 80%;
      height: 80%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 15px;
    }
    .form-field {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
    }
    label {
      margin-bottom: 5px;
    }
    input {
      padding: 8px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }
    button {
      padding: 10px 15px;
      border: none;
      background-color: #2196F3;
      color: white;
      border-radius: 4px;
      cursor: pointer;
    }
    button:hover {
      background-color: #1976D2;
    }
  `;
}

customElements.define('teachable-machine-upload-view', TeachableMachineUploadView);

export { TeachableMachineUploadView };
