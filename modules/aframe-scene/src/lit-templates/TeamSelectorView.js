import { LitElement, html, css } from 'lit';

export class TeamSelectorView extends LitElement {
    static properties = {
        currentTeamId: { type: String },
        teamName: { type: String },
        availableTeams: { type: Array }, // To hold the list of teams from SessionManager
        sessionManager: { type: Object }, // To hold the reference to SessionManager
    };

    constructor() {
        super();
        this.currentTeamId = null;
        this.teamName = '';
        this.availableTeams = [];
        this.sessionManager = null;
    }

    // Lifecycle callback for when the element is added to the DOM
    connectedCallback() {
        super.connectedCallback();

        // The SessionManager is now passed as a property, so we don't need to query the DOM.

        if (this.sessionManager) {
            console.log('SessionManager found:', this.sessionManager);
            // Initialize properties from SessionManager
            this.availableTeams = this.sessionManager.teams || [];
            // Use SessionManager's selectedTeam if available, otherwise check localStorage
            if (this.sessionManager.selectedTeam) {
                this.currentTeamId = this.sessionManager.selectedTeam.id;
                this.teamName = this.sessionManager.selectedTeam.name;
            } else {
                // Fallback to localStorage if SessionManager has no selected team
                const storedTeamId = localStorage.getItem('teamId');
                const storedTeamName = localStorage.getItem('teamName');
                if (storedTeamId && storedTeamName) {
                    this.currentTeamId = storedTeamId;
                    this.teamName = storedTeamName;
                }
            }

            // Listen for updates from SessionManager
            this.sessionManager.addEventListener('session-updated', this._handleSessionUpdate);
        } else {
            console.error('SessionManager element not found in DOM!');
            // Fallback: load directly from localStorage if SessionManager is not found
            const storedTeamId = localStorage.getItem('teamId');
            const storedTeamName = localStorage.getItem('teamName');
            if (storedTeamId && storedTeamName) {
                this.currentTeamId = storedTeamId;
                this.teamName = storedTeamName;
            }
        }
    }

    // Lifecycle callback for when the element is removed from the DOM
    disconnectedCallback() {
        super.disconnectedCallback();
        if (this.sessionManager) {
            this.sessionManager.removeEventListener('session-updated', this._handleSessionUpdate);
        }
    }

    // Handler for updates from SessionManager
    _handleSessionUpdate = (event) => {
        const { teams, selectedTeam } = event.detail;
        if (teams !== undefined) { // Check for undefined to allow empty arrays
            this.availableTeams = teams;
        }
        if (selectedTeam !== undefined) { // Check for undefined to allow null
            this.currentTeamId = selectedTeam ? selectedTeam.id : null;
            this.teamName = selectedTeam ? selectedTeam.name : '';
            // Update localStorage if SessionManager updates selected team
            if (selectedTeam) {
                localStorage.setItem('teamId', selectedTeam.id);
                localStorage.setItem('teamName', selectedTeam.name);
            } else {
                localStorage.removeItem('teamId');
                localStorage.removeItem('teamName');
            }
        }
    }

    // Handle team selection via button click
    selectTeam(teamId, teamName) {
        let deviceId = localStorage.getItem("deviceId");
        // Dispatch a custom event to inform SessionManager
        this.dispatchEvent(new CustomEvent('team-selected', {
            detail: { deviceId: deviceId, teamId: teamId, teamName: teamName },
            bubbles: true,
            composed: true
        }));

        // Update local state for immediate UI feedback
        this.currentTeamId = teamId;
        this.teamName = teamName;
        // localStorage is updated by SessionManager via _handleTeamSelected -> _updateAndNotify
    }

    // Render the view based on the current state
    render() {
        // Display current team name if teamId and teamName are set
        if (this.currentTeamId && this.teamName) {
            return html`
                <div class="team-selector-container">
                    <h2>Welcome to Team ${this.teamName}</h2>
                    <p>Team ID: ${this.currentTeamId}</p>
                    <button @click=${this._clearTeamSelection}>Change Team</button>
                </div>
            `;
        } else {
            // Display buttons to select a team if no team is selected
            return html`
                <div class="team-selector-container">
                    <h1>Select a Team</h1>
                    ${this.availableTeams.length === 0
                        ? html`<p>Loading teams...</p>`
                        : html`
                            ${this.availableTeams.map(team => html`
                                <button @click=${() => this.selectTeam(team.id, team.name)}>
                                    ${team.name}
                                </button>
                            `)}
                        `}
                </div>
            `;
        }
    }

    _clearTeamSelection() {
        this.currentTeamId = null;
        this.teamName = '';
        localStorage.removeItem('teamId');
        localStorage.removeItem('teamName');
        // Dispatch an event to inform SessionManager to clear its selection
        this.dispatchEvent(new CustomEvent('team-cleared', {
            bubbles: true,
            composed: true
        }));
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
