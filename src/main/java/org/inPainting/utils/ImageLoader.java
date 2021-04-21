package org.inPainting.utils;

import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.inPainting.nn.dataSets.ImageDataSetIterator;
import org.inPainting.nn.dataSets.ImageFileDataSetIterator;
import org.inPainting.nn.dataSets.ImageMemoryDataSetIterator;
import org.inPainting.nn.dataSets.preProcessors.GrayDataPreProcessor;

import java.io.File;

public final class ImageLoader {

    private WritableImage writableTemp;

    public WritableImage emptyImage(Color color, int width, int height) {

        writableTemp = new WritableImage(width, height);
        PixelWriter pixelWriter = writableTemp.getPixelWriter();

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixelWriter.setColor(x, y, color);

        return writableTemp;
    }

    public WritableImage drawImage(INDArray data, int width, int height) {

        writableTemp = new WritableImage(width, height);
        PixelWriter pixelWriter = writableTemp.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double r = data.getDouble(0,0,y,x);
                double g = data.getDouble(0,1,y,x);
                double b = data.getDouble(0,2,y,x);
                Color color = new Color(r, g, b, 1);
                pixelWriter.setColor(x, y, color);
            }
        }
        return writableTemp;
    }

    public static INDArray mergeImagesByMask(INDArray IImage, INDArray Mask, INDArray OImage, int width, int height) {

        INDArray indArray = OImage.dup();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (Mask.getDouble(0,0,y,x) == 0.0D) {
                    indArray.putScalar(new int[]{0,0,y,x},IImage.getDouble(0,0,y,x));
                    indArray.putScalar(new int[]{0,1,y,x},IImage.getDouble(0,1,y,x));
                    indArray.putScalar(new int[]{0,2,y,x},IImage.getDouble(0,2,y,x));
                }
            }
        }
        return indArray;
    }

    public ImageMemoryDataSetIterator prepareInMemoryData() {

        ImageDataSetIterator.FileEntry[] entries = new ImageDataSetIterator.FileEntry[new File("./data/256/expected/").listFiles().length];

        for (int i = 1; i < entries.length + 1; i++) {
            entries[i-1] = new ImageDataSetIterator.FileEntry(
                    new File("./data/256/inputs/input" + i + ".png"),
                    new File("./data/256/inputs/input" + i + "_mask.png"),
                    new File("./data/256/expected/expected" + i + ".png")
            );
        }
        return new ImageMemoryDataSetIterator(10, entries);
    }

    public ImageFileDataSetIterator prepareInFileData(){

        ImageFileDataSetIterator.FileEntry[] entries = new ImageFileDataSetIterator.FileEntry[new File("./data/256/expected/").listFiles().length];

        for (int i = 1; i < entries.length + 1; i++) {
            entries[i-1] = new ImageFileDataSetIterator.FileEntry(
                    new File("./data/256/inputs/input" + i + ".png"),
                    new File("./data/256/inputs/input" + i + "_mask.png"),
                    new File("./data/256/expected/expected" + i + ".png")
            );
        }
        return new ImageFileDataSetIterator(10, entries, null);
    }
}