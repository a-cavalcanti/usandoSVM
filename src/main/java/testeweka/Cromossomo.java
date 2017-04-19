package testeweka;

public class Cromossomo {
	
	private int svmType;
	private int cacheSize;
	private double cost;
	private double eps;
	private double gamma;
	private int kernel;
	private double nu;
	private boolean probability;
	private double loss;
	
	public double getLoss() {
		return loss;
	}
	public void setLoss(double loss) {
		this.loss = loss;
	}
	public int getSvmType() {
		return svmType;
	}
	public void setSvmType(int svmType) {
		this.svmType = svmType;
	}
	public int getCacheSize() {
		return cacheSize;
	}
	public void setCacheSize(int cacheSize) {
		this.cacheSize = cacheSize;
	}
	public double getCost() {
		return cost;
	}
	public void setCost(double cost) {
		this.cost = cost;
	}
	public double getEps() {
		return eps;
	}
	public void setEps(double eps) {
		this.eps = eps;
	}
	public double getGamma() {
		return gamma;
	}
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}
	public int getKernel() {
		return kernel;
	}
	public void setKernel(int kernel) {
		this.kernel = kernel;
	}
	public double getNu() {
		return nu;
	}
	public void setNu(double nu) {
		this.nu = nu;
	}
	public boolean getProbability() {
		return probability;
	}
	public void setProbability(boolean probability) {
		this.probability = probability;
	}
	
	
}
