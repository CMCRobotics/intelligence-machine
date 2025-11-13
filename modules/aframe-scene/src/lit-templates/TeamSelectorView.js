import { LitElement, html, css } from 'lit';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit'; // Assuming this is available


export class TeamSelectorView extends LitElement {
    static properties = {
        teams: { type: Array }, // Array of { id: string, name: string } objects
        activeViewName: { type: String }, // To potentially receive from ViewManager if needed
        deviceId: { type: String }, // To publish team selection
        currentTeamId: { type: String }, // To store the selected team ID
        teamName: { type: String } // To display the selected team name
    };

    constructor() {
        super();
        this.teams = [];
        this.currentTeamId = null;
        this.teamName = '';
        this.homieObserver = null;

        setLogLevel('debug');
    }

    // Lifecycle callback for when the element is added to the DOM
    connectedCallback() {
        super.connectedCallback();
        this.initTeamSelection();
        // Check localStorage immediately on connection
        this.currentTeamId = localStorage.getItem('teamId');
        if (this.currentTeamId) {
            // If teamId is found, we'll try to display it once teams are loaded
            // For now, we just store it. The render method will handle conditional display.
        }
    }

    // Lifecycle callback for when the element is removed from the DOM
    disconnectedCallback() {
        super.disconnectedCallback();
        if (this.homieObserver) {
            // Clean up subscription if necessary
            // Note: createMqttHomieObserver might manage its own lifecycle,
            // but explicit cleanup is good practice if possible.
            // For now, we assume it's managed or doesn't need explicit unsubscribe here.
            // this.homieObserver.disconnect();
        }
    }

    // Initialize MQTT observer and subscriptions
    async initTeamSelection() {
        try {
            // Assuming the MQTT broker is accessible at ws://localhost:9001
            // This URL might need to be configurable or passed as a property
            
            
            if (! this.homieObserver) {
                this.homieObserver = createMqttHomieObserver("ws://localhost:9001");
                
                this.homieObserver.updated$.subscribe(
                    (event) => {
                        if (event.type === 'property' 
                             && event.device.id.startsWith('team-') 
                             && event.node.id === 'info'
                             && event.property.id === 'name') {
                            const topicParts = event.topic.split('/');
                            // Expected topic: homie/team-XYZ/info/name
                            if (topicParts.length >= 3 && topicParts[1].startsWith('team-')) {
                                const teamId = topicParts[1];
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
                                if (this.currentTeamId === teamId) {
                                    this.teamName = teamName;
                                }
                            }
                        }
                    },
                    (error) => {
                        console.error('Error in Homie subscription:', error);
                    }
                );

                // Subscribe to team names
                this.homieObserver.subscribe("+/info/name");
            } else {
                console.error('Failed to create Homie observer.');
            }
        } catch (error) {
            console.error('Error initializing team selection:', error);
        }
    }

    // Handle team selection via button click
    selectTeam(teamId, teamName) {
        localStorage.setItem('teamId', teamId);
        this.currentTeamId = teamId;
        this.teamName = teamName;

        // Publish the selected team ID if deviceId is available
        if (this.deviceId) {
            // Assuming 'terminal' is a known device type or part of the session context
            // The topic format might need adjustment based on actual system
            const publishTopic = `terminal-${this.deviceId}/team-id/set`;
            this.homieObserver?.publish(publishTopic, teamId);
        }

        // Dispatch a custom event to notify the parent (ViewManager) that a team has been selected
        this.dispatchEvent(new CustomEvent('team-selected', {
            detail: { teamId: teamId, teamName: teamName },
            bubbles: true,
            composed: true
        }));
    }

    // Render the view based on the current state
    render() {
        const selectedTeam = this.teams.find(team => team.id === this.currentTeamId);

        if (this.currentTeamId && selectedTeam) {
            // Display current team name if teamId is set and team is found
            return html`
                <div class="team-selector-container">
                    <h2>Current Team: ${selectedTeam.name}</h2>
                    <p>Team ID: ${this.currentTeamId}</p>
                </div>
            `;
        } else {
            // Display buttons to select a team if no team is selected or found
            return html`
                <div class="team-selector-container">
                    <h1>Select a Team</h1>
                    ${this.teams.length === 0
                        ? html`<p>Loading teams...</p>`
                        : html`
                            ${this.teams.map(team => html`
                                <button @click=${() => this.selectTeam(team.id, team.name)}>
                                    ${team.name}
                                </button>
                            `)}
                        `}
                </div>
            `;
        }
    }

    static styles = css`
        .team-selector-container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 20px;
            box-sizing: border-box;
            font-family: sans-serif;
        }
        h1, h2 {
            color: #333;
            margin-bottom: 20px;
        }
        button {
            margin: 10px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f0f0f0;
        }
        button:hover {
            background-color: #e0e0e0;
        }
        p {
            color: #666;
        }
    `;
}

// Define the custom element
customElements.define('team-selector-view', TeamSelectorView);

