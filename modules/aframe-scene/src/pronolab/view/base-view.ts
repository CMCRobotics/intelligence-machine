import { Session } from '../core/session';
import { logger } from '../logger';

export abstract class BaseView {
    protected container: HTMLElement;
    private testStatusElement: HTMLElement;
    protected session: Session;
    protected model: any;
    protected maxPredictions: number;
    protected modelURL: string | null = null;
    protected metadataURL: string | null = null;

    constructor(container: HTMLElement, session: Session) {
        this.container = container;
        this.session = session;
        this.maxPredictions = 0;
        this.testStatusElement = document.createElement('div');
        this.testStatusElement.className = 'test-status';
        this.container.appendChild(this.testStatusElement);

        this.session.onTestStatusChanged.subscribe(status => {
            this.testStatusElement.innerText = status;
            this.testStatusElement.style.display = status ? 'block' : 'none';
        });
    }

    public abstract init(): Promise<void>;
    public abstract show(): void;
    public abstract hide(): void;
    protected abstract loop(): Promise<void>;

    public setModel(modelURL: string, metadataURL: string) {
        this.modelURL = modelURL;
        this.metadataURL = metadataURL;
        this.loadModel();
    }

    protected abstract loadModel(): Promise<void>;

    // Helper to setup label container, common for views that display predictions
    protected setupLabelContainer(): void {
        // Clear previous content and create a new label container
        this.container.innerHTML = ''; // Clear existing content
        const labelContainer = document.createElement('div');
        labelContainer.id = 'label-container';
        this.container.appendChild(labelContainer);
    }

    // Helper to update labels, assuming labels are displayed in order
    protected updateLabel(index: number, text: string): void {
        const labelContainer = this.container.querySelector('#label-container');
        if (labelContainer) {
            // Ensure there's a div for this label index
            if (!labelContainer.children[index]) {
                const newLabel = document.createElement('div');
                labelContainer.appendChild(newLabel);
            }
            labelContainer.children[index].textContent = text;
        }
    }
}
