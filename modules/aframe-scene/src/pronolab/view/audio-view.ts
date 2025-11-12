import * as tmAudio from '@tensorflow-models/speech-commands';
import { Session } from '../core/session';
import { ModelTestBaseView } from './model-test-base-view'; // Changed import
import { logger } from '../logger';

export class AudioView extends ModelTestBaseView { // Changed inheritance
    // labelContainer is now inherited from ModelTestBaseView

    constructor(container: HTMLElement, session: Session) {
        super(container, session);
    }

    public async init(): Promise<void> {
        await super.init(); // Call super.init if it has logic
        // empty for now
    }

    public show(): void {
        super.setupLabelContainer(); // Setup the label container
        this.startListening();
    }

    public hide(): void {
        if (this.model) {
            this.model.stopListening();
        }
        super.hide(); // Call super.hide to clear container
    }

    protected async loadModel(): Promise<void> {
        if (this.modelURL && this.metadataURL) {
            logger.debug(`loading model from ${this.modelURL} and ${this.metadataURL}`);
            this.model = await tmAudio.create('BROWSER_FFT');
            this.maxPredictions = this.model.getTotalClasses();
            logger.debug(`model loaded with ${this.maxPredictions} classes`);
        }
    }

    protected async loop(): Promise<void> {
        // audio loop is handled by the library
    }

    private async startListening() {
        if (!this.model) {
            logger.error('model not loaded yet');
            return;
        }
        logger.debug('start listening');

        this.model.listen((result: { scores: { className: string, probability: number }[] }) => {
            const prediction = result.scores.map(s => ({ className: s.className, probability: s.probability }));
            this.session.onPrediction.next(prediction);
            // Use the inherited updateLabel method
            for (let i = 0; i < this.maxPredictions; i++) {
                const classPrediction =
                    result.scores[i].className + ": " + result.scores[i].probability.toFixed(2);
                this.updateLabel(i, classPrediction);
            }
        });
    }

    // Metadata logic remains specific to AudioView
    private get metadata(): { words: string[] } {
        const metadata: { words: string[] } = {
            words: []
        };
        try {
            const url = new URL(this.metadataURL!);
            const words = url.searchParams.get('words');
            if (words) {
                metadata.words = words.split(',');
            }
        } catch (e) {
            logger.error(e);
        }
        return metadata;
    }
}
