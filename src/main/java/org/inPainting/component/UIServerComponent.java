package org.inPainting.component;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.springframework.stereotype.Component;

import java.io.File;

@Component
public class UIServerComponent {

    private UIServer uiServer;
    private StatsStorage statsStorage;
    private StatsListener statsListener;
    private ComputationGraph currentNetwork;

    public UIServerComponent(){
        statsStorage = new FileStatsStorage(new File("ui-stats.dat"));
        statsListener = new StatsListener(statsStorage);
        uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);
        uiServer.enableRemoteListener();
    }

    private boolean useUI = true;

    public void reinitialize(ComputationGraph multiLayerNetwork) {
        if (useUI) {
            if (currentNetwork != null)
                currentNetwork.getListeners().remove(statsListener);
            if (multiLayerNetwork != null)
                multiLayerNetwork.addListeners(statsListener, new PerformanceListener(1000,true));

            currentNetwork = multiLayerNetwork;
            System.gc();
        }
    }


    public void stop() {
        if (useUI) {
            uiServer.stop();
        }
    }
}
