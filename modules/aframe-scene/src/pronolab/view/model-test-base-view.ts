import { Session } from '../core/session';
import { BaseView } from './base-view';
import { logger } from '../logger';

export abstract class ModelTestBaseView extends BaseView {
    // labelContainer is now inherited from BaseView
    protected modelURL: string | null = null;
    protected metadataURL: string | null = null;
    // protected model: any; // Removed redundant declaration
    // protected maxPredictions: number; // This is declared in BaseView, but ModelTestBaseView needs to initialize it.

    constructor(container: HTMLElement, session: Session) {
        super(container, session);
        // The BaseView constructor already initializes maxPredictions to 0.
        // If ModelTestBaseView needs a different default or specific initialization, it should be done here.
    }

    public async init(): Promise<void> {
        // Load the model when the view is initialized
        await this.loadModel();
    }

    public show(): void {
        // Setup the label container before showing the view
        super.setupLabelContainer();
        // The actual display logic (e.g., starting webcam/listening) will be in subclasses
    }

    public hide(): void {
        // Subclasses should handle stopping any ongoing processes (e.g., webcam, listening)
        // super.hide(); // Removed incorrect call to abstract method
    }

    protected abstract loadModel(): Promise<void>;
    protected abstract loop(): Promise<void>; // Keep loop abstract as subclasses implement it

    // setModel is inherited from BaseView and calls loadModel
    // No need to override unless specific logic is required here
}
