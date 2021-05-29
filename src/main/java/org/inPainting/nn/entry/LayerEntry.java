package org.inPainting.nn.entry;


import org.deeplearning4j.nn.conf.layers.Layer;

public class LayerEntry extends LEntry {
    private final Layer layer;

    public LayerEntry(String layerName, Layer layer, String... inputs) {
        super(layerName, inputs, false);
        this.layer = layer;
    }

    public Layer getLayer() {
        return layer;
    }
}
