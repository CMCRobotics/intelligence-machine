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
    overallTimeout: { type: Number },
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
    this.uploadedModels = {}; // Tracks { modelName: [{ terminalId, teamId }] }
    this.selectedModel = this.models.length > 0 ? this.models[0].modelName : '';
    this.confidence = 80;
    this.duration = 1000;
    this.overallTimeout = 10000;
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
            if (event.node && event.node.id.startsWith('model-') && event.property && event.property.id === 'terminalId') {
              const teamId = event.device.id;
              const modelName = event.node.id.substring('model-'.length);
              const terminalId = event.property.value;

              if (!this.uploadedModels[modelName]) {
                this.uploadedModels[modelName] = [];
              }
              const existing = this.uploadedModels[modelName].find(e => e.terminalId === terminalId);
              if (!existing) {
                this.uploadedModels[modelName].push({ terminalId, teamId });
                console.log(`Registered upload of model '${modelName}' from terminal '${terminalId}' on team '${teamId}'`);
              }
            }
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
      // First, switch the view
      this.homieObserver.publish(`${terminalId}/ui-control/switch`, 'teachable-machine-upload');
  
      // Then, after a short delay, send the model data
      setTimeout(() => {
        this.homieObserver.publish(`${terminalId}/model-upload/name`, selectedModelData.modelName);
        this.homieObserver.publish(`${terminalId}/model-upload/type`, selectedModelData.modelType);
      }, 200);
    }
  }

  switchToWaitingView() {
    for (const terminalId of Object.keys(this.terminals)) {
      this.homieObserver.publish(`${terminalId}/ui-control/switch`, 'waiting-view');
    }
  }

  setupTest() {
    const selectedModelData = this.models.find(model => model.modelName === this.selectedModel);
    if (!selectedModelData) {
      console.error('Selected model data not found.');
      return;
    }
  
    const uploadsToTrigger = this.uploadedModels[selectedModelData.modelName];
  
    if (!uploadsToTrigger || uploadsToTrigger.length === 0) {
      alert(`No team has uploaded the "${selectedModelData.modelName}" model yet.`);
      return;
    }
  
    for (const uploadInfo of uploadsToTrigger) {
      const { terminalId, teamId } = uploadInfo;
      
      // First, switch the view
      this.homieObserver.publish(`terminal-${terminalId}/ui-control/switch`, `teachable-machine-${selectedModelData.modelType}`);
      
      // Then, after a short delay, send the model data
      setTimeout(() => {
        const payload = {
          name: selectedModelData.modelName,
          uploaderTeamId: teamId,
          type: selectedModelData.modelType
        };
        this.homieObserver.publish(`terminal-${terminalId}/activeModel/set`, JSON.stringify(payload));
        console.log(`Setting up test for model '${selectedModelData.modelName}' on terminal '${terminalId}' (team: ${teamId})`);
      }, 200);
    }
  }

  runTest() {
    const selectedModelData = this.models.find(model => model.modelName === this.selectedModel);
    if (!selectedModelData) {
      console.error('Selected model data not found.');
      return;
    }

    const uploadsToTrigger = this.uploadedModels[selectedModelData.modelName];

    if (!uploadsToTrigger || uploadsToTrigger.length === 0) {
      alert(`No team has uploaded the "${selectedModelData.modelName}" model yet. Please set up the test first.`);
      return;
    }

    for (const uploadInfo of uploadsToTrigger) {
      const { terminalId } = uploadInfo;
      const payload = JSON.stringify({
        confidence: this.confidence,
        duration: this.duration,
        class: Math.floor(Math.random() * 3),
        overallTimeout: this.overallTimeout
      });
      this.homieObserver.publish(`terminal-${terminalId}/activeModel/test`, payload);
      console.log(`Running test for model '${selectedModelData.modelName}' on terminal '${terminalId}'`);
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
          <button @click=${this.switchToWaitingView}>Switch all to Waiting View</button>
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
          <label>
            Overall Timeout (ms):
            <input type="number" .value=${this.overallTimeout} @input=${(e) => this.overallTimeout = e.target.value}>
          </label>
          <button @click=${this.setupTest}>Setup Test on Terminals</button>
          <button @click=${this.runTest}>Run Test on All</button>
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
