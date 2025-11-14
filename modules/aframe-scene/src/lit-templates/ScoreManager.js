import { LitElement, html } from 'lit';
import { createMqttHomieObserver } from '@cmcrobotics/homie-lit';
import {merge} from 'rxjs';

class ScoreManager extends LitElement {
  static properties = {
    scores: { type: Object },
  };

  constructor() {
    super();
    this.scores = {};
    this.terminalToTeamMap = {};
    this.homieObserver = null;
  }

  connectedCallback() {
    super.connectedCallback();
    this._initializeHomieObserver();
  }

  _initializeHomieObserver() {
    try {
      this.homieObserver = createMqttHomieObserver("ws://localhost:9001");

    merge(
      this.homieObserver.created$,
      this.homieObserver.updated$)
      .subscribe(
        (event) => {
          if (event.type === 'property') {
            if (event.property.id === 'testSuccess') {
              const deviceId = event.device.id;
              const teamId = this.terminalToTeamMap[deviceId];
              if (teamId) {
                this._incrementScore(teamId);
              }
            } else if (event.device.id.startsWith('terminal-') && event.node.id === 'info' && event.property.id === 'team') {
              const deviceId = event.device.id;
              const teamId = event.property.value;
              this.terminalToTeamMap[deviceId] = teamId;
              console.log(`Associated terminal ${deviceId} with team ${teamId}`);
            } else if (event.device.id.startsWith('team-') && event.node.id === 'info' && event.property.id === 'name') {
              const teamId = event.device.id;
              if (this.scores[teamId] === undefined) {
                this._initializeScore(teamId);
              }
            }
          }
        }
      );

      this.homieObserver.subscribe("+/info/team");
      this.homieObserver.subscribe("+/info/name");
      this.homieObserver.subscribe("+/+/testSuccess");
      console.log('ScoreManager initialized and subscribed to team, terminal, and testSuccess events');

    } catch (error) {
      console.error('Error initializing Homie observer for ScoreManager:', error);
    }
  }

  _initializeScore(teamId) {
    const newScores = { ...this.scores };
    newScores[teamId] = 0;
    this.scores = newScores;
    this._publishScore(teamId, 0);
    console.log(`Initialized score for team ${teamId}`);
  }

  _publishScore(teamId, score) {
    if (this.homieObserver) {
      const topic = `${teamId}/info/score`;
      this.homieObserver.publish(topic, score.toString(), { retain: true });
      console.log(`Published score for ${teamId}: ${score}`);
    }
  }

  _incrementScore(teamId) {
    const newScores = { ...this.scores };
    if (newScores[teamId] === undefined) {
      newScores[teamId] = 0;
    }
    newScores[teamId]++;
    this.scores = newScores;
    this._publishScore(teamId, newScores[teamId]);
    console.log(`Score updated: ${JSON.stringify(this.scores)}`);
    this.dispatchEvent(new CustomEvent('score-updated', {
        detail: { scores: this.scores },
        bubbles: true,
        composed: true
    }));
  }

  render() {
    return html`
      <style>
        :host {
          display: none;
        }
      </style>
    `;
  }
}

customElements.define('score-manager', ScoreManager);
