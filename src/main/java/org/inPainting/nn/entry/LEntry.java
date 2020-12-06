package org.inPainting.nn.entry;

public abstract class LEntry {
    protected String layerName;
    protected String[] inputs;
    protected boolean isVertex;

    public LEntry(String layerName, String[] inputs, boolean isVertex){
        this.layerName = layerName;
        this.inputs = inputs;
        this.isVertex = isVertex;
    }

    public String getLayerName() {
        return layerName;
    }

    public boolean isVertex() {
        return isVertex;
    }

    public String[] getInputs() {
        return inputs;
    }
}
