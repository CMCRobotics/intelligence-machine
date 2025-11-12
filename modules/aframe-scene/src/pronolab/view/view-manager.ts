import { Session } from '../core/session';
import { BaseView } from './base-view';
import { logger } from '../logger';

export class ViewManager {
    private container: HTMLElement;
    private session: Session;
    private views: { [key: string]: BaseView } = {};
    private activeView: BaseView | null = null;
    private modelURL: string | null = null;
    private metadataURL: string | null = null;
    private deviceId: string | null = null;

    constructor(container: HTMLElement, session: Session) {
        this.container = container;
        this.session = session;
        this.deviceId = localStorage.getItem('deviceId');
    }

    public addView(name: string, view: BaseView) {
        this.views[name] = view;
    }

    public init() {
        if (!this.deviceId) {
            logger.error('Device ID not found');
            return;
        }

        this.session.topic(`homie/terminal-${this.deviceId}/ui-control/model-type/set`)
            .subscribe((message: string) => this.setActiveView(message));

        this.session.topic(`homie/terminal-${this.deviceId}/ui-control/model-url/set`)
            .subscribe((message: string) => {
                this.modelURL = message;
                this.setModel();
            });

        this.session.topic(`homie/terminal-${this.deviceId}/ui-control/metadata-url/set`)
            .subscribe((message: string) => {
                this.metadataURL = message;
                this.setModel();
            });

        this.session.topic(`homie/terminal-${this.deviceId}/ui-control/switch/set`)
            .subscribe((message: string) => {
                if (this.activeView) {
                    if (message === 'model-test') {
                        this.activeView.show();
                    } else {
                        this.activeView.hide();
                    }
                }
            });

        this.session.topic(`homie/terminal-${this.deviceId}/ui-control/model-test/set`)
            .subscribe((message: string) => {
                if (this.activeView) {
                    logger.info(`[ModelTest] received test request: ${message}`);
                    try {
                        const { className, minConfidence, duration } = JSON.parse(message);
                        logger.info(`[ModelTest] triggering test for class "${className}" with min confidence ${minConfidence} for ${duration}ms`);
                        this.session.testModel(className, minConfidence, duration);
                    } catch (e) {
                        logger.error(`[ModelTest] failed to parse test request: ${e}`);
                    }
                }
            });
    }

    private setActiveView(name: string) {
        if (this.activeView) {
            this.activeView.hide();
        }
        this.activeView = this.views[name];
        if (this.activeView) {
            this.container.style.display = 'block';
            this.activeView.init();
            this.activeView.show();
            if (name !== 'upload-model' && name !== 'session-controller') {
                this.setModel();
            }
        }
    }

    private setModel() {
        if (this.activeView && this.modelURL && this.metadataURL) {
            this.activeView.setModel(this.modelURL, this.metadataURL);
        }
    }
}
