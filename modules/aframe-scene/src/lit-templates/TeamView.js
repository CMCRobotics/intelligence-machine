import { LitElement, html, css } from 'lit';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit';
import { merge } from 'rxjs';

class TeamView extends LitElement {
  static properties = {
    teams: { type: Object },
  };

  constructor() {
    super();
    this.teams = {};
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const mqttUrl = (window.APP_CONFIG && window.APP_CONFIG.MQTT_BROKER_URL) || `${scheme}://${window.location.hostname}:9001`;
    this.homieObserver = createMqttHomieObserver(mqttUrl);
    setLogLevel('debug');
  }

  connectedCallback() {
    super.connectedCallback();
    this.updateTimer = null;
    merge(this.homieObserver.created$, this.homieObserver.updated$).subscribe(
      (event) => {
        if (event.device && event.device.id.startsWith('team-')) {
          const teamId = event.device.id;
          if (!this.teams[teamId]) {
            this.teams[teamId] = { models: {} };
          }

          if (event.type === 'property' && event.node.id==="info" && event.property.id === 'name') {
            this.teams[teamId].name = event.property.value;
          }

          if (event.type === 'property' && event.node.id==="info" && event.property.id === 'score') {
            this.teams[teamId].score = event.property.value;
          }

          if (event.node && event.node.id.startsWith('model-')) {
            const modelId = event.node.id;
            if (!this.teams[teamId].models[modelId]) {
              this.teams[teamId].models[modelId] = {};
            }
            if (event.type === 'property') {
              this.teams[teamId].models[modelId][event.property.id] = event.property.value;
            }
          }
          if (this.updateTimer) {
            clearTimeout(this.updateTimer);
          }
          this.updateTimer = setTimeout(() => this.requestUpdate(), 0);
        }
      }
    );
    this.homieObserver.subscribe('homie/#');
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.updateTimer) {
      clearTimeout(this.updateTimer);
    }
  }

  _handleResetScores() {
    if (confirm('Are you sure you want to reset all scores? This action cannot be undone.')) {
        const scoreManager = document.querySelector('score-manager');
        if (scoreManager) {
            scoreManager.resetAllScores();
        } else {
            console.error('ScoreManager component not found.');
        }
    }
  }

  _handleIncrementScore(teamId) {
    const scoreManager = document.querySelector('score-manager');
    if (scoreManager) {
        scoreManager._incrementScore(teamId);
    } else {
        console.error('ScoreManager component not found.');
    }
  }

  render() {
    return html`
      <div class="team-list">
        <h1>Teams</h1>
        <button @click="${this._handleResetScores}" class="reset-button">Reset All Scores</button>
        ${Object.entries(this.teams).map(
          ([teamId, team]) => html`
            <div class="team-container">
              <h2>
                <span class="team-color" style="background-color: ${team.name ? team.name.toLowerCase() : 'grey'}"></span>
                 ${team.name} - Score: ${team.score || 0}
                 <button @click="${() => this._handleIncrementScore(teamId)}" class="increment-button">+1</button>
              </h2>
              <table>
                <thead>
                  <tr>
                    <th>Model Name</th>
                    <th>Model Type</th>
                    <th>Uploaded by</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  ${Object.values(team.models).map(
                    (model) => html`
                      <tr>
                        <td>${model.modelName}</td>
                        <td>${model.type}</td>
                        <td>${model.terminalId}</td>
                        <td>${new Date(model.timestamp).toLocaleString()}</td>
                      </tr>
                    `
                  )}
                </tbody>
              </table>
            </div>
          `
        )}
      </div>
    `;
  }

  static styles = css`
    :host {
      color: white;
    }
    .team-list {
      background-color: rgba(0, 0, 0, 0.5);
      padding: 20px;
      border-radius: 10px;
      width: 80%;
      max-width: 1200px;
    }
    .team-container {
      margin-bottom: 20px;
    }
    .team-color {
      display: inline-block;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      margin-right: 10px;
      vertical-align: middle;
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
    }
    th {
      background-color: #4CAF50;
      color: white;
    }
    .reset-button {
        background-color: #f44336;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .reset-button:hover {
        background-color: #da190b;
    }
    .increment-button {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        margin-left: 10px;
        vertical-align: middle;
    }
    .increment-button:hover {
        background-color: #45a049;
    }
  `;
}

customElements.define('team-view', TeamView);

export { TeamView };
