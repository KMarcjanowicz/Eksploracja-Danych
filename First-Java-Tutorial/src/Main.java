import java.util.*;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        //TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        // to see how IntelliJ IDEA suggests fixing it.
        System.out.print("Hello and welcome!\n");

        System.out.print("Max value of int= " + Integer.MAX_VALUE + "\n");
        System.out.print("Min value of int= " + Integer.MIN_VALUE + "\n");


        for (int i = 1; i <= 5; i++) {
            //TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint
            // for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
            System.out.println("i = " + i);
        }

        Printable printLambda = (s) -> System.out.println("Lambda" + s);

        List<Cat> lista = new ArrayList<>();
        lista.add(new Cat("Nati"));
        lista.add(new Cat("Punia"));
        lista.add(new Cat("Harvi"));

        PrintList(lista);
    }

    public static <Whatever> void Shout (Whatever quip){
        System.out.print("Shout: " + quip + "!!!\n");
    }

    private static void PrintList (List<Cat> list){
        for (Cat o : list) {
            o.print("Hey") ;
        }
    }
}

