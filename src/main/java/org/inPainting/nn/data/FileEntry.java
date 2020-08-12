package org.inPainting.nn.data;

import lombok.Getter;

import java.io.File;

public class FileEntry {

    @Getter
    private File input;
    @Getter
    private File output;
    @Getter
    private File mask;

    public FileEntry(File input, File output, File mask){
        this.input = input;
        this.output = output;
        this.mask = mask;
    }

    @Override
    protected void finalize() throws Throwable {
        input = null;
        output = null;
        mask = null;
    }
}
