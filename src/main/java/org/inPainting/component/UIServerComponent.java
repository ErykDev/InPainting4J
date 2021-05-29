package org.inPainting.component;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.springframework.stereotype.Component;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;

import java.io.File;

@Component
public class UIServerComponent {

    private final UIServer uiServer;
    private final StatsListener statsListener;
    private ComputationGraph currentNetwork;

    public UIServerComponent() {
        StatsStorage statsStorage = new FileStatsStorage(new File("ui-stats.dat"));
        statsListener = new StatsListener(statsStorage);
        uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);
        uiServer.enableRemoteListener();
    }

    private final boolean useUI = true;

    public void reinitialize(ComputationGraph multiLayerNetwork) {
        if (useUI) {
            if (currentNetwork != null)
                currentNetwork.getListeners().remove(statsListener);
            if (multiLayerNetwork != null)
                multiLayerNetwork.addListeners(statsListener);

            currentNetwork = multiLayerNetwork;
        }
    }

    public void stop() throws InterruptedException {
        if (useUI)
            uiServer.stop();
    }
}
