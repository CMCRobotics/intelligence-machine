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
    this.homieObserver = createMqttHomieObserver('ws://localhost:9001');
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

  render() {
    return html`
      <div class="team-list">
        <h1>Teams</h1>
        ${Object.values(this.teams).map(
          (team) => html`
            <div class="team-container">
              <h2>
                <span class="team-color" style="background-color: ${team.name ? team.name.toLowerCase() : 'grey'}"></span>
                 ${team.name} - Score: ${team.score || 0}
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
  `;
}

customElements.define('team-view', TeamView);

export { TeamView };
