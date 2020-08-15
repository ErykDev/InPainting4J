package org.inPainting.nn.data;

import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public abstract class ImageDataSetIterator implements MultiDataSetIterator {
    public abstract MultiDataSet nextRandom();
    public abstract void shuffle();
}
