import { LitElement, html, css } from 'lit';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit';
import { merge } from 'rxjs';

class TerminalView extends LitElement {
  static properties = {
    terminals: { type: Object },
    teams: { type: Object },
  };

  constructor() {
    super();
    this.terminals = {};
    this.teams = {};
    this.homieObserver = createMqttHomieObserver('ws://localhost:9001');
    setLogLevel('debug');
  }

  connectedCallback() {
    super.connectedCallback();
    merge(this.homieObserver.created$, this.homieObserver.updated$).subscribe(
      (event) => {
        if (event.device) {
          if (event.device.id.startsWith('terminal-')) {
            const terminalId = event.device.id;
            if (!this.terminals[terminalId]) {
              this.terminals[terminalId] = {};
            }
            if (event.type === 'property' && event.node.id === 'ui-control' && event.property.id === 'switch') {
              this.terminals[terminalId].view = event.property.value;
            }
            if (event.type === 'property' && event.node.id === 'info' && event.property.id === 'team') {
              this.terminals[terminalId].teamId = event.property.value;
            }
            this.requestUpdate();
          } else if (event.device.id.startsWith('team-')) {
            const teamId = event.device.id;
            if (!this.teams[teamId]) {
              this.teams[teamId] = {};
            }
            if (event.type === 'property' && event.device.id.startsWith('team-') 
                && event.node.id === 'info'
                && event.property.id === 'name') {
                  this.teams[teamId].name = event.property.value;
                  this.teams[teamId].color = event.property.value.toLowerCase();
            }
            
            this.requestUpdate();
          }
        }
      }
    );
    this.homieObserver.subscribe('homie/#');
  }

  render() {
    return html`
      <div class="device-list">
        <h1>Terminal Devices</h1>
        <table>
          <thead>
            <tr>
              <th>Device ID</th>
              <th>Current View</th>
              <th>Team Name</th>
              <th>Team Color</th>
            </tr>
          </thead>
          <tbody>
            ${Object.entries(this.terminals).map(
              ([terminalId, terminal]) => html`
                <tr>
                  <td>${terminalId.replace('terminal-', '')}</td>
                  <td>${terminal.view}</td>
                  <td>${terminal.teamId && this.teams[terminal.teamId] ? this.teams[terminal.teamId].name : 'N/A'}</td>
                  <td style="background-color: ${terminal.teamId && this.teams[terminal.teamId] ? this.teams[terminal.teamId].color : 'transparent'}"></td>
                </tr>
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
    .device-list {
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

customElements.define('terminal-view', TerminalView);

export { TerminalView };
