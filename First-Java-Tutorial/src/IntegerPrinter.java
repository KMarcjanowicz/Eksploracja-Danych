public class IntegerPrinter {

    Integer thingToPrint;

    public IntegerPrinter(Integer thing){
        this.thingToPrint = thing;
    }

    public void print(){
        System.out.print("Integer: " + thingToPrint);
    }
}
