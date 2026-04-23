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
    this.pendingIncrements = {};
  }

  connectedCallback() {
    super.connectedCallback();
    this._initializeHomieObserver();
  }

  _initializeHomieObserver() {
    try {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const mqttUrl = (window.APP_CONFIG && window.APP_CONFIG.MQTT_BROKER_URL) || `${scheme}://${window.location.hostname}:9001`;
      this.homieObserver = createMqttHomieObserver(mqttUrl);

    this.homieObserver.created$.subscribe((event) => {
        if (event.type === 'property') {
            if (event.device.id.startsWith('team-') && event.node.id === 'info' && event.property.id === 'score') {
                const teamId = event.device.id;
                const score = parseInt(event.property.value, 10);
                if (!isNaN(score)) {
                    const newScores = { ...this.scores };
                    newScores[teamId] = score;
                    this.scores = newScores;
                    console.log(`Initialized score for team ${teamId}: ${score}`);
                    this._applyPendingIncrements(teamId);
                }
            } else if (event.device.id.startsWith('terminal-') && event.node.id === 'info' && event.property.id === 'team') {
                const deviceId = event.device.id;
                const teamId = event.property.value;
                this.terminalToTeamMap[deviceId] = teamId;
                console.log(`Associated terminal ${deviceId} with team ${teamId}`);
            }
        }
    });

    this.homieObserver.updated$.subscribe((event) => {
        if (event.type === 'property') {
            if (event.property.id === 'testSuccess') {
                const deviceId = event.device.id;
                const teamId = this.terminalToTeamMap[deviceId];
                if (teamId) {
                    this._incrementScore(teamId);
                }
            }
        }
    });

      this.homieObserver.subscribe("+/info/team");
      this.homieObserver.subscribe("+/info/name");
      this.homieObserver.subscribe("+/+/testSuccess");
      this.homieObserver.subscribe('+/info/score');
      console.log('ScoreManager initialized and subscribed to team, terminal, and testSuccess events');

    } catch (error) {
      console.error('Error initializing Homie observer for ScoreManager:', error);
    }
  }

  resetAllScores() {
    const newScores = { ...this.scores };
    const teamIds = Object.keys(newScores);
    teamIds.forEach(teamId => {
        newScores[teamId] = 0;
        this._publishScore(teamId, 0);
    });
    this.scores = newScores;
    console.log('All scores have been reset to 0.');
  }

  _applyPendingIncrements(teamId) {
    if (this.pendingIncrements[teamId]) {
        const count = this.pendingIncrements[teamId];
        console.log(`Applying ${count} pending increments for ${teamId}`);
        const newScores = { ...this.scores };
        newScores[teamId] += count;
        this.scores = newScores;
        this._publishScore(teamId, newScores[teamId]);
        delete this.pendingIncrements[teamId];
    }
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
    if (typeof newScores[teamId] === 'number') {
        newScores[teamId]++;
        this.scores = newScores;
        this._publishScore(teamId, newScores[teamId]);
        console.log(`Score updated: ${JSON.stringify(this.scores)}`);
        this.dispatchEvent(new CustomEvent('score-updated', {
            detail: { scores: this.scores },
            bubbles: true,
            composed: true
        }));
    } else {
        this.pendingIncrements[teamId] = (this.pendingIncrements[teamId] || 0) + 1;
        console.log(`Buffered increment for ${teamId}. Total pending: ${this.pendingIncrements[teamId]}`);
    }
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
