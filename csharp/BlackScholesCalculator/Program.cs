// csharp/BlackScholesCalculator/Program.cs

using System;

namespace BlackScholesCalculator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("--- Black-Scholes Option Pricer (C#) ---");

            // Get inputs from user
            double S = GetDoubleInput("Enter Current Underlying Price (e.g., 100): ");
            double K = GetDoubleInput("Enter Option Strike Price (e.g., 105): ");
            double T = GetDoubleInput("Enter Time to Expiration (Years, e.g., 0.5): ", allowZero: false, allowNegative: false);
            double r = GetDoubleInput("Enter Risk-Free Interest Rate (Annualized, e.g., 0.05): ");
            double sigma = GetDoubleInput("Enter Volatility (Annualized, e.g., 0.2): ", allowZero: false, allowNegative: false);
            string optionType = GetOptionTypeInput("Enter Option Type ('call' or 'put'): ");

            // Calculate Price
            double price = 0;
            if (optionType == "call")
            {
                price = BlackScholesModel.CalculateCallPrice(S, K, T, r, sigma);
            }
            else // put
            {
                price = BlackScholesModel.CalculatePutPrice(S, K, T, r, sigma);
            }

            // Calculate Greeks
            double delta = BlackScholesModel.CalculateDelta(S, K, T, r, sigma, optionType);
            double gamma = BlackScholesModel.CalculateGamma(S, K, T, r, sigma);
            double vega = BlackScholesModel.CalculateVega(S, K, T, r, sigma);
            double theta = BlackScholesModel.CalculateTheta(S, K, T, r, sigma, optionType);
            double rho = BlackScholesModel.CalculateRho(S, K, T, r, sigma, optionType);


            // Display Results
            Console.WriteLine("\n--- Calculation Results ---");
            Console.WriteLine($"Option Type:       {optionType.Substring(0, 1).ToUpper() + optionType.Substring(1)}"); // Capitalize
            Console.WriteLine($"Underlying Price:  {S,15:F4}");
            Console.WriteLine($"Strike Price:      {K,15:F4}");
            Console.WriteLine($"Time to Expiry(Y): {T,15:F4}");
            Console.WriteLine($"Risk-Free Rate:    {r,15:F4} ({r * 100:F2}%)");
            Console.WriteLine($"Volatility:        {sigma,15:F4} ({sigma * 100:F2}%)");
            Console.WriteLine("-----------------------------------------");
            Console.WriteLine($"Theoretical Price: {price,15:F4}");
            Console.WriteLine("-----------------------------------------");
            Console.WriteLine("Greeks:");
            Console.WriteLine($"  Delta: {delta,18:F6}");
            Console.WriteLine($"  Gamma: {gamma,18:F6}");
            Console.WriteLine($"  Vega:  {vega,18:F6} (per 1% vol change)");
            Console.WriteLine($"  Theta: {theta,18:F6} (per day)");
            Console.WriteLine($"  Rho:   {rho,18:F6} (per 1% rate change)");
            Console.WriteLine("-----------------------------------------");

            Console.WriteLine("\nPress Enter to exit.");
            Console.ReadLine();
        }

        // Helper function to get validated double input
        static double GetDoubleInput(string prompt, bool allowZero = true, bool allowNegative = true)
        {
            double value;
            while (true)
            {
                Console.Write(prompt);
                string input = Console.ReadLine() ?? "";
                if (double.TryParse(input, out value))
                {
                    if (!allowZero && value == 0) {
                         Console.WriteLine("Value cannot be zero. Please try again.");
                    } else if (!allowNegative && value < 0) {
                         Console.WriteLine("Value cannot be negative. Please try again.");
                    } else {
                         return value;
                    }
                }
                else
                {
                    Console.WriteLine("Invalid input. Please enter a valid number.");
                }
            }
        }

        // Helper function to get validated option type input
        static string GetOptionTypeInput(string prompt)
        {
             while (true)
             {
                 Console.Write(prompt);
                 string input = (Console.ReadLine() ?? "").Trim().ToLower();
                 if (input == "call" || input == "put")
                 {
                     return input;
                 }
                 else
                 {
                     Console.WriteLine("Invalid input. Please enter 'call' or 'put'.");
                 }
             }
        }
    }
}