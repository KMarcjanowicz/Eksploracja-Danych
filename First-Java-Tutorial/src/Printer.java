import java.io.Serial;
import java.io.Serializable;

public class Printer <GenericType extends Serializable & Printable> {

    GenericType GenericThingToPrint;

    public Printer (GenericType thing){
        this.GenericThingToPrint = thing;
    }

    public void print(){
        Main.Shout(GenericThingToPrint);
        System.out.print("Printer: " + GenericThingToPrint + "\n");
    }
}
