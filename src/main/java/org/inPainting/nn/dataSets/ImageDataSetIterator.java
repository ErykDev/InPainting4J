package org.inPainting.nn.dataSets;

import javafx.scene.image.Image;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.File;
import java.io.IOException;

public abstract class ImageDataSetIterator implements MultiDataSetIterator {
    @Getter
    public long maxSize;

    /**
     * @returns Random MultiDataSet from the set
     * */
    public abstract MultiDataSet nextRandom();

    /**
     * shuffling the MultiDataSetIterator
     * */
    public abstract void shuffle();

    protected abstract INDArray convertToRank4INDArrayOutput(Image inputImage);
    protected abstract INDArray convertToRank4INDArrayInput(Image inputImage);
    protected abstract INDArray convertToRank4INDArrayInputMask(Image inputImageMask);

    protected abstract MultiDataSet convertToDataSet(FileEntry fileEntry) throws IOException;
    public abstract MultiDataSet next();

    protected double scaleColor(double value) {
        return (value);
    }

    public static class FileEntry {
        @Getter
        private final File input;

        @Getter
        private final File input_mask;

        @Getter
        private final File output;

        public FileEntry(File input, File input_mask, File output){
            this.input = input;
            this.output = output;
            this.input_mask = input_mask;
        }
    }
}
