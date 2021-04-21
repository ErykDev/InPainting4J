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

    public abstract MultiDataSet nextRandom();
    public abstract void shuffle();

    protected abstract INDArray convertToRank4INDArrayOutput(Image inputImage);
    protected abstract INDArray convertToRank4INDArrayInput(Image inputImage);

    protected abstract MultiDataSet convertToDataSet(FileEntry fileEntry) throws IOException;
    public abstract MultiDataSet next();

    protected double scaleColor(double value) {
        return (value);
    }

    public static class FileEntry {
        @Getter
        private File input;

        @Getter
        private File output;

        public FileEntry(File input, File output){
            this.input = input;
            this.output = output;
        }
    }
}
