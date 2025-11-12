import * as tmImage from '@teachablemachine/image';
import { Session } from '../core/session';
import { ModelTestBaseView } from './model-test-base-view'; // Changed import
import { logger } from '../logger';

export class ImageView extends ModelTestBaseView { // Changed inheritance
    private webcam: any;
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
        this.initWebcam();
    }

    public hide(): void {
        if (this.webcam) {
            this.webcam.stop();
        }
        super.hide(); // Call super.hide to clear container
    }

    protected async loadModel(): Promise<void> {
        if (this.modelURL && this.metadataURL) {
            logger.debug(`loading model from ${this.modelURL} and ${this.metadataURL}`);
            this.model = await tmImage.load(this.modelURL, this.metadataURL);
            this.maxPredictions = this.model.getTotalClasses();
            logger.debug(`model loaded with ${this.maxPredictions} classes`);
        }
    }

    protected async loop(): Promise<void> {
        this.webcam.update(); // update the webcam frame
        await this.predict();
        window.requestAnimationFrame(() => this.loop());
    }

    private async initWebcam() {
        if (!this.model) {
            logger.error('model not loaded yet');
            return;
        }
        logger.debug('initializing webcam');
        const flip = true; // whether to flip the webcam
        this.webcam = new tmImage.Webcam(200, 200, flip); // width, height, flip
        await this.webcam.setup(); // request access to the webcam

        // append elements to the DOM before playing
        this.container.appendChild(this.webcam.canvas);
        
        await this.webcam.play();
        window.requestAnimationFrame(() => this.loop());
    }

    private async predict() {
        // predict can take in an image, video or canvas html element
        const prediction = await this.model.predict(this.webcam.canvas);
        this.session.onPrediction.next(prediction);
        // Use the inherited updateLabel method
        for (let i = 0; i < this.maxPredictions; i++) {
            const classPrediction =
                prediction[i].className + ": " + prediction[i].probability.toFixed(2);
            this.updateLabel(i, classPrediction);
        }
    }
}
