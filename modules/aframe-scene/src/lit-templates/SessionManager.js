import { LitElement, html } from 'lit';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit'; // Using HomieObserver as requested

class SessionManager extends LitElement {
  static properties = {
    teams: { type: Array },
    selectedTeam: { type: Object },
    homieObserver: { type: Object },
    mqttConnected: { type: Boolean },
  };

  constructor() {
    super();
    this.teams = [];
    this.selectedTeam = null;
    this.homieObserver = null;
    this.mqttConnected = false;
    
    // setLogLevel('debug'); // Set log level for HomieObserver
    this._loadTeamsFromLocalStorage();
    this._updateAndNotify(); // Notify initial state
  }

  connectedCallback() {
    super.connectedCallback();
    this._initializeHomieObserver();
    this.addEventListener('team-selected', this._handleTeamSelected);
    this.addEventListener('team-cleared', this._handleTeamCleared);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.homieObserver) {
      // Clean up subscription if necessary
      // HomieObserver might manage its own lifecycle, but explicit cleanup is good practice.
      // For now, we assume it's managed or doesn't need explicit unsubscribe here.
      // this.homieObserver.disconnect(); // If such a method exists
    }
    this.removeEventListener('team-selected', this._handleTeamSelected);
    this.removeEventListener('team-cleared', this._handleTeamCleared);
  }

  _initializeHomieObserver() {
    try {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
      this.homieObserver = createMqttHomieObserver(`${scheme}://${window.location.hostname}:9001`);
      this.mqttConnected = true; // Assume connected upon creation for now

      this.homieObserver.created$.subscribe(
        (event) => {
          if (event.type === 'property' 
               && event.device.id.startsWith('team-') 
               && event.node.id === 'info'
               && event.property.id === 'name') {
                const teamId = event.device.id;
                const teamName = event.property.value;

                // Update or add team to the list
                const existingTeamIndex = this.teams.findIndex(team => team.id === teamId);
                if (existingTeamIndex === -1) {
                    this.teams = [...this.teams, { id: teamId, name: teamName }];
                } else {
                    // Update existing team name if it changed
                    const updatedTeams = [...this.teams];
                    updatedTeams[existingTeamIndex] = { id: teamId, name: teamName };
                    this.teams = updatedTeams;
                }

                // If a teamId was previously selected and now we have its name, update teamName
                if (this.selectedTeam && this.selectedTeam.id === teamId) {
                    this.selectedTeam = { id: teamId, name: teamName };
                }
                this._updateAndNotify(); // Notify listeners after data update
          }
        },
        (error) => {
            console.error('Error in Homie subscription:', error);
            this.mqttConnected = false; // Mark as disconnected on error
            this._updateAndNotify();
        }
      );

      // Subscribe to team names
      this.homieObserver.subscribe("+/info/name");
      console.log('HomieObserver initialized and subscribed to +/info/name');

    } catch (error) {
        console.error('Error initializing Homie observer:', error);
        this.mqttConnected = false; // Mark as disconnected on error
    }
  }

  _loadTeamsFromLocalStorage() {
    try {
      const storedTeams = localStorage.getItem('teams');
      if (storedTeams) {
        this.teams = JSON.parse(storedTeams);
      }
      const storedSelectedTeam = localStorage.getItem('selectedTeam');
      if (storedSelectedTeam) {
        this.selectedTeam = JSON.parse(storedSelectedTeam);
      }
    } catch (error) {
      console.error('Error loading from localStorage:', error);
    }
  }

  _saveTeamsToLocalStorage() {
    try {
      localStorage.setItem('teams', JSON.stringify(this.teams));
      localStorage.setItem('selectedTeam', JSON.stringify(this.selectedTeam));
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }

  _updateAndNotify() {
    this._saveTeamsToLocalStorage();
    this.dispatchEvent(new CustomEvent('session-updated', {
        detail: {
            teams: this.teams,
            selectedTeam: this.selectedTeam
        },
        bubbles: true,
        composed: true
    }));
  }

  _handleTeamSelected(event) {
    const { deviceId, teamId, teamName } = event.detail;
    const newTeam = { id: teamId, name: teamName };
    console.log('Team selected event received:', newTeam);

    this.selectedTeam = newTeam;
    // Add team to list if it doesn't exist
    if (!this.teams.some(t => t.id === teamId)) {
      this.teams = [...this.teams, newTeam];
    }
    
    this._updateAndNotify(); // Save and notify listeners
    this._publishTeamSelectionToMqtt(deviceId,newTeam);
  }

  _handleTeamCleared() {
    console.log('Team cleared event received.');
    this.selectedTeam = null;
    // Optionally clear teams list if needed, or keep it
    // this.teams = []; 
    this._updateAndNotify(); // Save and notify listeners
  }

  _publishTeamSelectionToMqtt(deviceId,team) {
    if (this.homieObserver && this.mqttConnected) {
      console.log(`Publishing team selection to MQTT: ${JSON.stringify(team)}`);
      // Publish the selected team ID
      // The topic format might need adjustment based on actual system
      // Assuming 'terminal' is a known device type or part of the session context
      // and deviceId is available or can be inferred.
      // For now, we'll use a placeholder topic.
      const publishTopic = `terminal-${deviceId}/info/team`; // Example topic
      this.homieObserver.publish(publishTopic, team.id, {retain:true});
    } else {
        console.warn('MQTT not connected or HomieObserver not available. Cannot publish tea m selection.');
    }
  }

  render() {
    // SessionManager might not render anything itself, or it could render status indicators
    return html`
      <style>
        :host {
          display: none; /* Typically, session managers don't render UI */
        }
      </style>
    `;
  }
}

customElements.define('session-manager', SessionManager);
