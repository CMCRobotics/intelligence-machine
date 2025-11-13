import { LitElement, html, css } from 'lit';
import { createMqttHomieObserver, setLogLevel } from '@cmcrobotics/homie-lit'; // Assuming this is available
import { Subject } from 'rxjs'; // Import Subject for RxJS
import { filter, bufferTime, tap } from 'rxjs/operators'; // Import RxJS operators

export class SessionManager extends LitElement {
    static properties = {
        teams: { type: Array }, // Array of { id: string, name: string } objects
        currentTeamId: { type: String }, // To store the selected team ID
        teamName: { type: String }, // To display the selected team name
        deviceId: { type: String } // To publish team selection
    };

    constructor() {
        super();
        this.teams = [];
        this.currentTeamId = null;
        this.teamName = '';
        this.deviceId = null; // Will be set externally or via property
        this.homieObserver = null;
        this._teamsReady = false; // Flag to track if teams are ready

        // Subject to manage the stream of team updates
        this.teamUpdateSubject = new Subject();

        // setLogLevel('debug'); // Keep log level for debugging
    }

    // Lifecycle callback for when the element is added to the DOM
    connectedCallback() {
        super.connectedCallback();
        this.initSession();
        // Check localStorage immediately on connection
        this.currentTeamId = localStorage.getItem('teamId');
        if (this.currentTeamId) {
            // If teamId is found, we'll try to display it once teams are loaded
            // For now, we just store it.
        }
    }

    // Lifecycle callback for when the element is removed from the DOM
    disconnectedCallback() {
        super.disconnectedCallback();
        if (this.homieObserver) {
            // Clean up subscription if necessary
            // this.homieObserver.disconnect(); // Uncomment if disconnect method exists and is needed
        }
        // Complete the subject to clean up RxJS resources
        this.teamUpdateSubject.complete();
    }

    // Initialize MQTT observer and subscriptions
    async initSession() {
        try {
            // Assuming the MQTT broker is accessible at ws://localhost:9001
            // This URL might need to be configurable or passed as a property
            if (!this.homieObserver) {
                this.homieObserver = createMqttHomieObserver("ws://localhost:9001");

                // Filter for team name updates and push them to the subject
                this.homieObserver.updated$.pipe(
                    filter(event =>
                        event.type === 'property' &&
                        event.device.id.startsWith('team-') &&
                        event.node.id === 'info' &&
                        event.property.id === 'name'
                    ),
                    tap(event => {
                        // Process individual updates immediately for selection/update events
                        const topicParts = event.topic.split('/');
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
                                // Dispatch event when team name is updated for a selected team
                                this.dispatchEvent(new CustomEvent('session-team-updated', {
                                    detail: { teamId: this.currentTeamId, teamName: this.teamName },
                                    bubbles: true,
                                    composed: true
                                }));
                            }
                        }
                    }),
                    // Buffer updates to bunch them
                    bufferTime(200)
                ).subscribe(
                    (bufferedEvents) => {
                        // This block will execute after a 500ms pause in team name updates
                        // or after the first batch of updates if they arrive quickly.
                        // We re-process the teams list here to ensure it's up-to-date
                        // and then signal that teams are ready.
                        
                        // Re-apply all updates from the buffer to ensure consistency
                        // (This might be redundant if tap already updated this.teams,
                        // but ensures the state is consistent with the buffered batch)
                        bufferedEvents.forEach(event => {
                            const topicParts = event.topic.split('/');
                            if (topicParts.length >= 3 && topicParts[1].startsWith('team-')) {
                                const teamId = topicParts[1];
                                const teamName = event.property.value;
                                const existingTeamIndex = this.teams.findIndex(team => team.id === teamId);
                                if (existingTeamIndex === -1) {
                                    this.teams = [...this.teams, { id: teamId, name: teamName }];
                                } else {
                                    const updatedTeams = [...this.teams];
                                    updatedTeams[existingTeamIndex] = { id: teamId, name: teamName };
                                    this.teams = updatedTeams;
                                }
                            }
                        });

                        if (!this._teamsReady && this.teams.length > 0) { // Only dispatch if not ready and we have teams
                            this._teamsReady = true;
                            console.info('Dispatching teams-ready event', this.teams);
                            this.dispatchEvent(new CustomEvent('session-teams-ready', {
                                detail: { teams: this.teams },
                                bubbles: true,
                                composed: true
                            }));
                        }
                    },
                    (error) => {
                        console.error('Error in RxJS stream for team updates:', error);
                    }
                );

                // Subscribe to team names
                this.homieObserver.subscribe("+/info/name");
            } else {
                console.error('Failed to create Homie observer.');
            }
        } catch (error) {
            console.error('Error initializing session:', error);
        }
    }

    // Handle team selection
    selectTeam(teamId, teamName) {
        localStorage.setItem('teamId', teamId);
        this.currentTeamId = teamId;
        this.teamName = teamName;

        // Publish the selected team ID if deviceId is available
        if (this.deviceId && this.homieObserver) {
            // Assuming 'terminal' is a known device type or part of the session context
            // The topic format might need adjustment based on actual system
            const publishTopic = `terminal-${this.deviceId}/team-id/set`;
            this.homieObserver.publish(publishTopic, teamId);
        }

        // Dispatch a custom event to notify other components that a team has been selected
        this.dispatchEvent(new CustomEvent('session-team-selected', {
            detail: { teamId: teamId, teamName: teamName },
            bubbles: true,
            composed: true
        }));
    }

    // Method to set the device ID, which might be needed for publishing
    setDeviceId(id) {
        this.deviceId = id;
    }

    // Method to get the current team details
    getCurrentTeam() {
        return {
            id: this.currentTeamId,
            name: this.teamName
        };
    }

    // Method to get all available teams
    getTeams() {
        return this.teams;
    }

    // Render method is not strictly necessary for a manager element,
    // but can be useful for debugging or if it has its own UI.
    // For now, we'll keep it minimal.
    render() {
        // This element is intended to manage state, not render UI directly.
        // It might render a hidden div or nothing at all.
        // For debugging, we could render something:
        return html`
            <div style="display: none;">Session Manager</div>
        `;
    }

    static styles = css`
        :host {
            display: block; /* Make it a block element */
        }
    `;
}

// Define the custom element
customElements.define('session-manager', SessionManager);
