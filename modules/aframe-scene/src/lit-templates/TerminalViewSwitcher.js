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
    this.models = [
      { modelName: 'greeter', modelType: 'image' },
      { modelName: 'sign-language', modelType: 'image' },
      { modelName: 'speak-to-me', modelType: 'speech' },
      { modelName: 'strike-a-pose', modelType: 'pose' }
    ];
    this.selectedModel = this.models.length > 0 ? this.models[0].modelName : '';
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
          }
        }
      }
    );
    this.homieObserver.subscribe('homie/#');
  }

  switchToUploadView() {
    const selectedModelData = this.models.find(model => model.modelName === this.selectedModel);
    if (!selectedModelData) return;

    for (const terminalId of Object.keys(this.terminals)) {
      this.homieObserver.publish(`${terminalId}/ui-control/switch`, 'teachable-machine-upload');
      this.homieObserver.publish(`${terminalId}/model-upload/name`, selectedModelData.modelName);
      this.homieObserver.publish(`${terminalId}/model-upload/type`, selectedModelData.modelType);
    }
  }

  triggerTest() {
    const selectedModelData = this.models.find(model => model.modelName === this.selectedModel);
    if (!selectedModelData) return;

    for (const terminalId of Object.keys(this.terminals)) {
      this.homieObserver.publish(`${terminalId}/ui-control/switch`, `teachable-machine-${selectedModelData.modelType}`);
      this.homieObserver.publish(`${terminalId}/activeModel/name`, selectedModelData.modelName);
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
          <select .value=${this.selectedModel} @change=${(e) => this.selectedModel = e.target.value}>
            ${this.models.map(model => html`<option value=${model.modelName}>${model.modelName}</option>`)}
          </select>
          <button @click=${this.switchToUploadView}>Switch All to Upload</button>
        </div>
        <div class="control-section">
          <h2>Teachable Model Test Trigger</h2>
          <select .value=${this.selectedModel} @change=${(e) => this.selectedModel = e.target.value}>
            ${this.models.map(model => html`<option value=${model.modelName}>${model.modelName}</option>`)}
          </select>
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
