public class Cat implements Printable {

    private final String name;
    public Cat(String _name){
        this.name = _name;
    }
    @Override
    public void print(String s) {
        System.out.println(this.name + " says: Meow!");
    }
}
