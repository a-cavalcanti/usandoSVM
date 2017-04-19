package testeweka;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.converters.ConverterUtils.DataSource;

public class UsandoWeka {

	public static void main(String[] args) throws Exception {
		
		int geracoes = 1000;
		ArrayList<Cromossomo> populacao = new ArrayList<Cromossomo>();
		populacao = gerarPopulacao(100);
		
		for (int i = 0; i < geracoes; i++) {			
			ArrayList<Double> pontuacao = avaliaPopulacao(populacao);
			ArrayList<Integer> posicoes = roleta(pontuacao);
			ArrayList<Cromossomo> novaPopulacao = realizaCrossOver(populacao, posicoes);
			novaPopulacao = realizaMutacao(novaPopulacao);
			ArrayList<Double> pontuacao2 = avaliaPopulacao(novaPopulacao);
			populacao = sobreviventes(populacao, novaPopulacao, pontuacao, pontuacao2);
		}
		

	}
	
	public static double gerarRandomIntervalo(){
		Random random = new Random();
		double valor = random.nextDouble();
		
		while (valor >= 0.10 || valor == 0.0) {
			valor = random.nextDouble();			
		}
		
		return valor;
	}
	
	public static ArrayList<Cromossomo> gerarPopulacao(int quantidade){
		
		ArrayList<Cromossomo> populacao = new ArrayList<Cromossomo>();
		Random random = new Random();
		
		for (int i = 0; i < quantidade; i++) {
			Cromossomo cromossomo = new Cromossomo();		
			cromossomo.setCacheSize(random.nextInt(1001));
			cromossomo.setCost(random.nextDouble());
			cromossomo.setEps(random.nextDouble());
			cromossomo.setGamma(random.nextDouble());
			cromossomo.setKernel(random.nextInt(4));
			cromossomo.setNu(gerarRandomIntervalo());
			cromossomo.setProbability(random.nextBoolean());
			cromossomo.setSvmType(random.nextInt(2));
			cromossomo.setLoss(random.nextDouble());
			populacao.add(cromossomo);
		}
		
		return populacao;		
	}
	
	public static ArrayList<Double> avaliaPopulacao(ArrayList<Cromossomo> populacao) throws Exception{
		
		ArrayList<Double> pontuacao = new ArrayList<Double>();
		
		DataSource dsTreino = new DataSource("src/main/java/testeweka/regressao-ptbr-treino-entailment.arff");
		DataSource dsTeste = new DataSource("src/main/java/testeweka/regressao-ptbr-treino-entailment.arff");
		
		Instances insTreino = dsTreino.getDataSet();
		Instances insTeste = dsTeste.getDataSet();		
		insTreino.setClassIndex(4);
		insTeste.setClassIndex(4);
		
		LibSVM lbsvm = new LibSVM();
		
		
		for (int i = 0; i < populacao.size(); i++) {
			
			int svmType=0, kernelType=0;
			
			if (populacao.get(i).getSvmType() == 0) {
				svmType = LibSVM.SVMTYPE_C_SVC;
			}else if(populacao.get(i).getSvmType() == 1) {
				svmType = LibSVM.SVMTYPE_NU_SVC;
			}
			
			/*
			else if(populacao.get(i).getSvmType() == 1) {
				svmType = LibSVM.SVMTYPE_EPSILON_SVR;
			}			
			else if(populacao.get(i).getSvmType() == 3) {
				svmType = LibSVM.SVMTYPE_NU_SVR;
			}
			*/
			
			if (populacao.get(i).getKernel() == 0){
				kernelType = LibSVM.KERNELTYPE_LINEAR;
			}else if (populacao.get(i).getKernel() == 1){
				kernelType = LibSVM.KERNELTYPE_POLYNOMIAL;
			}else if (populacao.get(i).getKernel() == 2){
				kernelType = LibSVM.KERNELTYPE_RBF;
			}else if (populacao.get(i).getKernel() == 3){
				kernelType = LibSVM.KERNELTYPE_SIGMOID;
			}
			
			double loss = populacao.get(i).getLoss();
			int cacheSize = populacao.get(i).getCacheSize();
			double eps = populacao.get(i).getEps();
			double gamma = populacao.get(i).getGamma();
			double nu = populacao.get(i).getNu();
			boolean probability = populacao.get(i).getProbability();
			double cost = populacao.get(i).getCost();
			
			lbsvm.setSVMType(new SelectedTag(svmType, LibSVM.TAGS_SVMTYPE));
			lbsvm.setKernelType(new SelectedTag(kernelType, LibSVM.TAGS_KERNELTYPE ));
			lbsvm.setLoss(loss);
			lbsvm.setCacheSize(cacheSize);
			lbsvm.setEps(eps);
			lbsvm.setGamma(gamma);			
			lbsvm.setNu(nu);
			lbsvm.setProbabilityEstimates(probability);
			lbsvm.setCost(cost);
			
			System.out.println(nu);
			
			lbsvm.buildClassifier(insTreino);
			
			Evaluation eval = new Evaluation(insTreino);
			eval.evaluateModel(lbsvm, insTeste);
			//System.out.println(eval.toSummaryString("\nResults\n======\n", true));
			//System.out.println(eval.toClassDetailsString());
			//System.out.println(eval.weightedPrecision());					
			//System.out.println(eval.weightedRecall());
			double fmeasure = eval.weightedFMeasure();
			System.out.println("F-measure = "+fmeasure);
			pontuacao.add(fmeasure);
			salvaResultados(svmType, kernelType, loss, cacheSize, eps, gamma, nu, probability, cost, fmeasure);
		}
		return pontuacao;
	}
	
	public static double round(double value, int places) {
	    if (places < 0) throw new IllegalArgumentException();

	    long factor = (long) Math.pow(10, places);
	    value = value * factor;
	    long tmp = Math.round(value);
	    return (double) tmp / factor;
	}
	
	public static void salvaResultados(int svmType, int kernelType, double loss, int cacheSize, double eps, double gamma, double nu, boolean probability, double cost, double fmeasure) throws IOException{
		File tabela8 = new File("./resultados.csv");
		FileWriter fw8 = new FileWriter(tabela8, true);
		BufferedWriter table8 = new BufferedWriter(fw8);
		
		table8.write("" +svmType+","
						+kernelType+","
						+loss+","
						+cacheSize+","
						+eps+","
						+gamma+","
						+nu+","
						+probability+","
						+cost+","
						+fmeasure+"\n");
		
		table8.close();
	}
	
	public static ArrayList<Integer> roleta(ArrayList<Double> pontuacao) {
		
		ArrayList<Double> valores = new ArrayList<Double>();
		double soma = 0.0;
		Random random = new Random();
		
		for (int i = 0; i < pontuacao.size(); i++) {
		 soma+= pontuacao.get(i);	
		}
		double elemento = 0.0;
		for (int i = 0; i < pontuacao.size(); i++) {
			elemento += pontuacao.get(i)/soma;			
			valores.add( elemento );
		}
				
		ArrayList<Integer> posicoes = new ArrayList<Integer>();
		
		for (int i = 0; i < pontuacao.size() ; i++) {				
			double sorteado = random.nextDouble();			
			
			int pos = 0;
			for (int j = 0; j < valores.size(); j++) {				
				if (sorteado > valores.get(j)) {
					pos = j;
				}
			}
			posicoes.add(pos);
		}
		return posicoes;
	}
	
	public static ArrayList<Cromossomo> realizaCrossOver(ArrayList<Cromossomo> populacao, ArrayList<Integer> posicoes){
		
		ArrayList<Cromossomo> novaPopulacao = new ArrayList<Cromossomo>();
		Random random = new Random();
		
		for (int i = 0; i < posicoes.size()-1; i=i+2) {
			
			Cromossomo filho1 = new Cromossomo();
			Cromossomo filho2 = new Cromossomo();			
			
			double valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setCacheSize(populacao.get(posicoes.get(i+1)).getCacheSize());
				filho2.setCacheSize(populacao.get(posicoes.get(i)).getCacheSize());
			}else{
				filho1.setCacheSize(populacao.get(posicoes.get(i)).getCacheSize());
				filho2.setCacheSize(populacao.get(posicoes.get(i+1)).getCacheSize());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setCost(populacao.get(posicoes.get(i+1)).getCost());
				filho2.setCost(populacao.get(posicoes.get(i)).getCost());
			}else{
				filho1.setCost(populacao.get(posicoes.get(i)).getCost());
				filho2.setCost(populacao.get(posicoes.get(i+1)).getCost());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setEps(populacao.get(posicoes.get(i+1)).getEps());
				filho2.setEps(populacao.get(posicoes.get(i)).getEps());
			}else{
				filho1.setEps(populacao.get(posicoes.get(i)).getEps());
				filho2.setEps(populacao.get(posicoes.get(i+1)).getEps());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setGamma(populacao.get(posicoes.get(i+1)).getGamma());
				filho2.setGamma(populacao.get(posicoes.get(i)).getGamma());
			}else{
				filho1.setGamma(populacao.get(posicoes.get(i)).getGamma());
				filho2.setGamma(populacao.get(posicoes.get(i+1)).getGamma());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setKernel(populacao.get(posicoes.get(i+1)).getKernel());
				filho2.setKernel(populacao.get(posicoes.get(i)).getKernel());
			}else{
				filho1.setKernel(populacao.get(posicoes.get(i)).getKernel());
				filho2.setKernel(populacao.get(posicoes.get(i+1)).getKernel());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setLoss(populacao.get(posicoes.get(i+1)).getLoss());
				filho2.setLoss(populacao.get(posicoes.get(i)).getLoss());
			}else{
				filho1.setLoss(populacao.get(posicoes.get(i)).getLoss());
				filho2.setLoss(populacao.get(posicoes.get(i+1)).getLoss());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setNu(populacao.get(posicoes.get(i+1)).getNu());
				filho2.setNu(populacao.get(posicoes.get(i)).getNu());
			}else{
				filho1.setNu(populacao.get(posicoes.get(i)).getNu());
				filho2.setNu(populacao.get(posicoes.get(i+1)).getNu());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setProbability(populacao.get(posicoes.get(i+1)).getProbability());
				filho2.setProbability(populacao.get(posicoes.get(i)).getProbability());
			}else{
				filho1.setProbability(populacao.get(posicoes.get(i)).getProbability());
				filho2.setProbability(populacao.get(posicoes.get(i+1)).getProbability());
			}
			
			valor = random.nextDouble();
			if (valor <= 0.10) {
				filho1.setSvmType(populacao.get(posicoes.get(i+1)).getSvmType());
				filho2.setSvmType(populacao.get(posicoes.get(i)).getSvmType());
			}else{
				filho1.setSvmType(populacao.get(posicoes.get(i)).getSvmType());
				filho2.setSvmType(populacao.get(posicoes.get(i+1)).getSvmType());
			}
			
			novaPopulacao.add(filho1);
			novaPopulacao.add(filho2);
		}//FOR
		
		return novaPopulacao;
	}
	
	public static ArrayList<Cromossomo> realizaMutacao(ArrayList<Cromossomo> novaPopulacao){
		
		Random random = new Random();
		
		for (int i = 0; i < novaPopulacao.size(); i++) {
			
			double trocar = random.nextDouble();
			
			if (trocar <= 0.10) {
				int valor = random.nextInt(9);
				
				switch (valor) {
				case 0:
					novaPopulacao.get(i).setCacheSize(random.nextInt(1001));
					break;
				case 1:
					novaPopulacao.get(i).setCost(random.nextDouble());
					break;
				case 2:
					novaPopulacao.get(i).setEps(random.nextDouble());
					break;
				case 3:
					novaPopulacao.get(i).setGamma(random.nextDouble());
					break;
				case 4:
					novaPopulacao.get(i).setKernel(random.nextInt(4));
					break;
				case 5:
					novaPopulacao.get(i).setLoss(random.nextDouble());
					break;
				case 6:
					novaPopulacao.get(i).setNu(gerarRandomIntervalo());
					break;
				case 7:
					novaPopulacao.get(i).setProbability(random.nextBoolean());
					break;
				case 8:
					novaPopulacao.get(i).setSvmType(random.nextInt(2));
					break;
				default:
					break;
				}
			}
			
		}
		
		return novaPopulacao;
	}
	
	public static ArrayList<Cromossomo> sobreviventes(ArrayList<Cromossomo> populacao, ArrayList<Cromossomo> novaPopulacao, ArrayList<Double> resultado1, ArrayList<Double> resultado2){
		
		double[] porcentagem = new double[resultado1.size() + resultado2.size()];
		ArrayList<Cromossomo> sobreviventes = new ArrayList<Cromossomo>();
		
		for (int i = 0; i < resultado1.size(); i++) {
			porcentagem[i] = resultado1.get(i);
		}
		
		for (int i = 0; i < resultado2.size(); i++) {
			porcentagem[i+resultado1.size()] = resultado2.get(i);
		}
		
		int maior = maiorPosicao(porcentagem);
		int[] vencedores = torneio(porcentagem, maior);
		
		ArrayList<Integer> winners = new ArrayList<Integer>();
		winners.add(maior);
		
		for (int i = 0; i < vencedores.length; i++) {
			winners.add(vencedores[i]);
		}
		
		for (int i = 0; i < winners.size(); i++) {
			int pos = winners.get(i);
			if (pos < resultado1.size()) {
				sobreviventes.add(populacao.get(pos));
			}else{
				int pos2 = pos - resultado1.size();
				sobreviventes.add(novaPopulacao.get(pos2) );
			}
		}		
		
		return sobreviventes;
	}
	
public static int[] torneio(double[] porcentagem, int maiorPosicao){
		
		Random random = new Random();
		int[] melhores = new int[(porcentagem.length/2)-1];
		ArrayList<Integer> sorteados = new ArrayList<Integer>();
		
		for (int i = 0; i < (porcentagem.length/2)-1; i++) {
			
			int s1 = random.nextInt(porcentagem.length);
			int s2 = random.nextInt(porcentagem.length);
			
			while(s1 == maiorPosicao || sorteados.contains(s1) ){
				s1 = random.nextInt(porcentagem.length);
			}
			sorteados.add(s1);			
			
			while (s2==maiorPosicao || s1 == s2 || sorteados.contains(s2)) {
				s2 = random.nextInt(porcentagem.length);				
			}
			sorteados.add(s2);
			
			if (porcentagem[s1] > porcentagem[s2]) {
				melhores[i] = s1;
				porcentagem[s1] = 0.0;
			}else{
				melhores[i] = s2;
				porcentagem[s2] = 0.0;
			}
			
			
		}
		
		
		return melhores;
		
	}
	
	public static int maiorPosicao(double[] porcentagem){
		
		double max = Double.MIN_VALUE;
		int pos =0;
		
		for (int i = 0; i < porcentagem.length; i++) {
			if (porcentagem[i] > max) {
				max = porcentagem[i];
				pos = i;
			}
		}
		
		return pos;
	}

}
