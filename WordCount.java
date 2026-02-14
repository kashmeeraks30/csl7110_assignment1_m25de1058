import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Iterator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

  public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
  
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    
    @Override
    public void map(LongWritable key, Text value, Context context)
    throws IOException, InterruptedException {

        String inputline = value.toString();
        
        inputline = inputline.replaceAll("[^0-9a-zA-Z\\s]", "");
        
        //inputline = inputline.toLowerCase();
        
        StringTokenizer tokenizer = new StringTokenizer(inputline);
        
        while (tokenizer.hasMoreTokens()) {
            word.set(tokenizer.nextToken());
            context.write(word, one);
        }
    }
  }

   public static class Reduce extends Reducer<Text,IntWritable,Text,IntWritable> {
   
    private IntWritable result = new IntWritable();

    @Override
    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
       int total = 0;
       Iterator<IntWritable> itr = values.iterator();
       while (itr.hasNext()) {
           total += itr.next().get();
       }
      result.set(total);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(WordCount.class);
    job.setMapperClass(Map.class);
    job.setCombinerClass(Reduce.class);
    job.setReducerClass(Reduce.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    job.getConfiguration().setLong("mapreduce.input.fileinputformat.split.maxsize", 134217728);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    long start_time=System.currentTimeMillis();
    boolean success=job.waitForCompletion(true);
    long end_time=System.currentTimeMillis();

    long exec_time=end_time-start_time;

    System.out.println("Execution time:" + exec_time + "milliseconds");
    System.exit(success ? 0 : 1);
  }
}
