import { LitElement, html, css } from 'lit';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit'; // Assuming this is available
import './SessionManager.js'; // Import the SessionManager


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
        // this.homieObserver = null; // Removed as it's now in SessionManager

        // setLogLevel('debug'); // Moved to SessionManager
    }

    // Lifecycle callback for when the element is added to the DOM
    connectedCallback() {
        super.connectedCallback();
        // this.initTeamSelection(); // Removed, SessionManager handles initialization

        // Check localStorage immediately on connection
        this.currentTeamId = localStorage.getItem('teamId');
        // If teamId is found, we'll try to display it once teams are loaded
        // For now, we just store it. The render method will handle conditional display.

        // Add event listeners for SessionManager events
        this.addEventListener('session-team-selected', this.handleTeamSelected);
        this.addEventListener('session-team-updated', this.handleTeamUpdated);
        this.addEventListener('session-teams-ready', this.handleTeamsReady); // Listen for the new event

        // Note: We no longer query for sessionManager here.
        // State updates will be handled by the event listeners.
        // If deviceId is available, it should ideally be set on SessionManager
        // when SessionManager is ready or rendered. For now, we assume it will be set.
    }

    // Lifecycle callback for when the element is removed from the DOM
    disconnectedCallback() {
        super.disconnectedCallback();
        // Remove event listeners to prevent memory leaks
        this.removeEventListener('session-team-selected', this.handleTeamSelected);
        this.removeEventListener('session-team-updated', this.handleTeamUpdated);
        this.removeEventListener('session-teams-ready', this.handleTeamsReady); // Remove listener for new event
    }

    // Handle team selection event from SessionManager
    handleTeamSelected(event) {
        const { teamId, teamName } = event.detail;
        this.currentTeamId = teamId;
        this.teamName = teamName;
        // The actual selection logic (localStorage, publish) is handled by SessionManager
        // We just update our local state to re-render.
        
        // Update teams list as well, in case it was not fully populated before selection
        const sessionManager = this.shadowRoot.querySelector('session-manager');
        if (sessionManager) {
            this.teams = sessionManager.getTeams();
        }
    }

    // Handle team update event from SessionManager (e.g., if team name changes)
    handleTeamUpdated(event) {
        const { teamId, teamName } = event.detail;
        if (this.currentTeamId === teamId) {
            this.teamName = teamName;
        }
        // Update the teams list as well, in case it changed
        const sessionManager = this.shadowRoot.querySelector('session-manager');
        if (sessionManager) {
            this.teams = sessionManager.getTeams();
        }
    }

    // Handle the event when SessionManager signals that teams are ready
    handleTeamsReady(event) {
        const { teams } = event.detail;
        this.teams = teams;

        // If a team was previously selected from localStorage, ensure its name is updated
        if (this.currentTeamId) {
            const selectedTeam = this.teams.find(team => team.id === this.currentTeamId);
            if (selectedTeam) {
                this.teamName = selectedTeam.name;
            }
        }
    }

    // Render the view based on the current state
    render() {
        // Get the SessionManager instance to access its state
        const sessionManager = this.shadowRoot.querySelector('session-manager');
        
        // If SessionManager is not yet available or not ready, show loading
        if (!sessionManager) {
            return html`
                <div class="team-selector-container">
                    <p>Loading session...</p>
                </div>
            `;
        }

        // Ensure deviceId is set on SessionManager if it's available
        // This might be called multiple times, but it's safe.
        if (this.deviceId && sessionManager.deviceId !== this.deviceId) {
            sessionManager.setDeviceId(this.deviceId);
        }

        // Use state from SessionManager
        const teams = sessionManager.getTeams();
        const currentTeamId = sessionManager.getCurrentTeam().id;
        const currentTeamName = sessionManager.getCurrentTeam().name;

        const selectedTeam = teams.find(team => team.id === currentTeamId);

        if (currentTeamId && selectedTeam) {
            // Display current team name if teamId is set and team is found
            return html`
                <div class="team-selector-container">
                    <h2>Current Team: ${selectedTeam.name}</h2>
                    <p>Team ID: ${currentTeamId}</p>
                </div>
            `;
        } else {
            // Display buttons to select a team if no team is selected or found
            return html`
                <div class="team-selector-container">
                    <h1>Select a Team</h1>
                    ${teams.length === 0
                        ? html`<p>Loading teams...</p>`
                        : html`
                            ${teams.map(team => html`
                                <button @click=${() => sessionManager.selectTeam(team.id, team.name)}>
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
