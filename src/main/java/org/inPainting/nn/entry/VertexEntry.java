package org.inPainting.nn.entry;

import org.deeplearning4j.nn.conf.graph.GraphVertex;


public class VertexEntry extends LEntry {

    private GraphVertex vertex;

    public VertexEntry(String layerName, GraphVertex reshapeVertex, String... inputs) {
        super(layerName, inputs, true);
        this.vertex = reshapeVertex;
    }

    public GraphVertex getVertex() {
        return vertex;
    }
}
