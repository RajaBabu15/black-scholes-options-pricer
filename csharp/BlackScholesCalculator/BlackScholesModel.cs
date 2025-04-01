// csharp/BlackScholesCalculator/BlackScholesModel.cs

using System;
// Import MathNet.Numerics for statistical functions (Normal CDF and PDF)
using MathNet.Numerics.Distributions;

namespace BlackScholesCalculator
{
    public static class BlackScholesModel
    {
        // N(x) - Cumulative Distribution Function (CDF) for standard normal distribution
        private static double N(double x)
        {
            return Normal.CDF(0, 1, x); // Mean 0, StdDev 1
        }

        // n(x) - Probability Density Function (PDF) for standard normal distribution
        private static double n(double x)
        {
            return Normal.PDF(0, 1, x); // Mean 0, StdDev 1
        }

        private static double CalculateD1(double S, double K, double T, double r, double sigma)
        {
            if (T <= 0 || sigma <= 0) // Handle edge cases
            {
                // Use a large number to approximate infinity for CDF calculations
                // Adjust sign based on S vs discounted K
                return (S > K * Math.Exp(-r * T)) ? 10.0 : -10.0;
            }
            return (Math.Log(S / K) + (r + 0.5 * Math.Pow(sigma, 2)) * T) / (sigma * Math.Sqrt(T));
        }

        private static double CalculateD2(double S, double K, double T, double r, double sigma)
        {
             if (T <= 0 || sigma <= 0) // Handle edge cases (must be consistent with d1)
            {
                 return (S > K * Math.Exp(-r * T)) ? 10.0 : -10.0;
            }
            return CalculateD1(S, K, T, r, sigma) - sigma * Math.Sqrt(T);
        }

        // --- Pricing Functions ---
        public static double CalculateCallPrice(double S, double K, double T, double r, double sigma)
        {
             if (T <= 0) return Math.Max(0.0, S - K);
             if (sigma <= 0) return Math.Max(0.0, S - K * Math.Exp(-r * T));

            double d1 = CalculateD1(S, K, T, r, sigma);
            double d2 = CalculateD2(S, K, T, r, sigma);
            return S * N(d1) - K * Math.Exp(-r * T) * N(d2);
        }

        public static double CalculatePutPrice(double S, double K, double T, double r, double sigma)
        {
             if (T <= 0) return Math.Max(0.0, K - S);
             if (sigma <= 0) return Math.Max(0.0, K * Math.Exp(-r * T) - S);

            double d1 = CalculateD1(S, K, T, r, sigma);
            double d2 = CalculateD2(S, K, T, r, sigma);
            return K * Math.Exp(-r * T) * N(-d2) - S * N(-d1);
        }

        // --- Greeks ---
        public static double CalculateDelta(double S, double K, double T, double r, double sigma, string optionType)
        {
             if (T <= 0 || sigma <= 0)
             {
                 if (optionType.ToLower() == "call") return (S >= K) ? 1.0 : 0.0;
                 else return (S <= K) ? -1.0 : 0.0;
             }

            double d1 = CalculateD1(S, K, T, r, sigma);
            if (optionType.ToLower() == "call")
            {
                return N(d1);
            }
            else if (optionType.ToLower() == "put")
            {
                return N(d1) - 1.0;
            }
            throw new ArgumentException("Invalid option type. Use 'call' or 'put'.");
        }

        public static double CalculateGamma(double S, double K, double T, double r, double sigma)
        {
             if (T <= 0 || sigma <= 0 || S <= 0) return 0.0; // Gamma is zero/undefined

            double d1 = CalculateD1(S, K, T, r, sigma);
            return n(d1) / (S * sigma * Math.Sqrt(T));
        }

        public static double CalculateVega(double S, double K, double T, double r, double sigma)
        {
             if (T <= 0 || sigma <= 0) return 0.0; // Vega is zero

            double d1 = CalculateD1(S, K, T, r, sigma);
            // Return Vega per 1% change in volatility
            return S * n(d1) * Math.Sqrt(T) * 0.01;
        }

         public static double CalculateTheta(double S, double K, double T, double r, double sigma, string optionType)
         {
              if (T <= 0 || sigma <= 0 || S <= 0) return 0.0; // Theta is zero/undefined

             double d1 = CalculateD1(S, K, T, r, sigma);
             double d2 = CalculateD2(S, K, T, r, sigma);

             double term1 = -(S * n(d1) * sigma) / (2 * Math.Sqrt(T));
             double term2;

             if (optionType.ToLower() == "call")
             {
                 term2 = r * K * Math.Exp(-r * T) * N(d2);
                 // Return Theta per day (negative sign convention indicates value decay)
                 return (term1 - term2) / 365.0;
             }
             else if (optionType.ToLower() == "put")
             {
                 term2 = r * K * Math.Exp(-r * T) * N(-d2);
                 // Return Theta per day
                 return (term1 + term2) / 365.0;
             }
             throw new ArgumentException("Invalid option type. Use 'call' or 'put'.");
         }

        public static double CalculateRho(double S, double K, double T, double r, double sigma, string optionType)
        {
             if (T <= 0) return 0.0; // Rho is zero at expiry

             double d2 = CalculateD2(S, K, T, r, sigma);
             double rhoVal;

             if (optionType.ToLower() == "call")
             {
                 rhoVal = K * T * Math.Exp(-r * T) * N(d2);
             }
             else if (optionType.ToLower() == "put")
             {
                 rhoVal = -K * T * Math.Exp(-r * T) * N(-d2);
             }
             else
             {
                throw new ArgumentException("Invalid option type. Use 'call' or 'put'.");
             }
             // Return Rho per 1% change in interest rate
             return rhoVal * 0.01;
        }
    }
}