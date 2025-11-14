import { LitElement, html, css } from 'lit';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit';
import { merge } from 'rxjs';

class TerminalViewSwitcher extends LitElement {
  static properties = {
    terminals: { type: Object },
    teams: { type: Object },
    models: { type: Array },
    selectedModel: { type: String },
    confidence: { type: Number },
    duration: { type: Number },
  };

  constructor() {
    super();
    this.terminals = {};
    this.teams = {};
    this.models = [];
    this.selectedModel = '';
    this.confidence = 80;
    this.duration = 1000;
    this.homieObserver = createMqttHomieObserver('ws://localhost:9001');
    setLogLevel('debug');
  }

  connectedCallback() {
    super.connectedCallback();
    merge(this.homieObserver.created$, this.homieObserver.updated$).subscribe(
      (event) => {
        if (event.device) {
          if (event.device.id.startsWith('terminal-')) {
            this.terminals[event.device.id] = event.device;
          } else if (event.device.id.startsWith('team-')) {
            const teamId = event.device.id;
            if (!this.teams[teamId]) {
              this.teams[teamId] = { models: {} };
            }
            if (event.node && event.node.id.startsWith('model-')) {
              const modelId = event.node.id;
              if (!this.teams[teamId].models[modelId]) {
                this.teams[teamId].models[modelId] = {};
              }
              if (event.type === 'property') {
                this.teams[teamId].models[modelId][event.property.id] = event.property.value;
              }
              this.updateModels();
            }
          }
        }
      }
    );
    this.homieObserver.subscribe('homie/#');
  }

  updateModels() {
    const models = [];
    for (const team of Object.values(this.teams)) {
      for (const model of Object.values(team.models)) {
        if (model.modelName) {
          models.push(model);
        }
      }
    }
    this.models = models;
    if (this.models.length > 0 && !this.selectedModel) {
      this.selectedModel = this.models[0].modelName;
    }
    this.requestUpdate();
  }

  switchToUploadView() {
    for (const terminalId of Object.keys(this.terminals)) {
      this.homieObserver.publish(`${terminalId}/ui-control/switch`, 'teachable-machine-upload');
      this.homieObserver.publish(`${terminalId}/model-upload/name`, this.selectedModel);
    }
  }

  triggerTest() {
    for (const terminalId of Object.keys(this.terminals)) {
      this.homieObserver.publish(`${terminalId}/ui-control/switch`, 'teachable-machine-image');
      const payload = JSON.stringify({
        confidence: this.confidence,
        duration: this.duration,
      });
      this.homieObserver.publish(`${terminalId}/activeModel/test`, payload);
    }
  }

  render() {
    return html`
      <div class="switcher-container">
        <div class="control-section">
          <h2>Teachable Model Upload</h2>
          <select @change=${(e) => this.selectedModel = e.target.value}>
            ${this.models.map(model => html`<option value=${model.modelName}>${model.modelName}</option>`)}
          </select>
          <button @click=${this.switchToUploadView}>Switch All to Upload</button>
        </div>
        <div class="control-section">
          <h2>Teachable Model Test Trigger</h2>
          <label>
            Confidence:
            <input type="number" .value=${this.confidence} @input=${(e) => this.confidence = e.target.value}>
          </label>
          <label>
            Duration (ms):
            <input type="number" .value=${this.duration} @input=${(e) => this.duration = e.target.value}>
          </label>
          <button @click=${this.triggerTest}>Trigger Test on All</button>
        </div>
      </div>
    `;
  }

  static styles = css`
    .switcher-container {
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding: 20px;
      background-color: rgba(0, 0, 0, 0.5);
      border-radius: 10px;
      color: white;
    }
    .control-section {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    select, input, button {
      padding: 8px;
      border-radius: 5px;
      border: 1px solid #ccc;
    }
    button {
      background-color: #4CAF50;
      color: white;
      cursor: pointer;
    }
  `;
}

customElements.define('terminal-view-switcher', TerminalViewSwitcher);

export { TerminalViewSwitcher };
