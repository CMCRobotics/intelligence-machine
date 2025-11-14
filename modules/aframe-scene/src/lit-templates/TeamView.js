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
        <table>
          <thead>
            <tr>
              <th>Color</th>
              <th>Team</th>
              <th>Model Name</th>
              <th>Model Type</th>
              <th>Uploaded by</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            ${Object.values(this.teams).map(
              (team) => html`
                ${Object.values(team.models).map(
                  (model) => html`
                    <tr>
                      <td style="background-color: ${team.name.toLowerCase()}">&nbsp;</td>
                      <td>${team.name}</td>
                      <td>${model.modelName}</td>
                      <td>${model.type}</td>
                      <td>${model.terminalId}</td>
                      <td>${new Date(model.timestamp).toLocaleString()}</td>
                    </tr>
                  `
                )}
              `
            )}
          </tbody>
        </table>
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
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 8px;
    }
    th {
      background-color: #4CAF50;
      color: white;
    }
  `;
}

customElements.define('team-view', TeamView);

export { TeamView };
