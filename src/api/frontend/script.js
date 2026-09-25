// ============================================================
// Form and UI Elements
// ============================================================

const form = document.getElementById("predictionForm");

const loadingMessage =
    document.getElementById("loadingMessage");

const errorMessage =
    document.getElementById("errorMessage");

const submitButton =
    form.querySelector("button[type='submit']");

const newPredictionButton =
    document.getElementById("newPredictionButton");

const resultBox =
    document.getElementById("result");


// ============================================================
// Prediction Form Submission
// ============================================================

form.addEventListener("submit", async function (event) {

    event.preventDefault();


    // ========================================================
    // Start Loading State
    // ========================================================

    loadingMessage.classList.remove("hidden");

    errorMessage.classList.add("hidden");

    submitButton.disabled = true;

    submitButton.textContent = "Analyzing...";


    // ========================================================
    // Collect Customer Data
    // ========================================================

    const customerData = {

        person_age: Number(
            document.getElementById("person_age").value
        ),

        person_income: Number(
            document.getElementById("person_income").value
        ),

        person_home_ownership:
            document.getElementById(
                "person_home_ownership"
            ).value,

        person_emp_length: Number(
            document.getElementById(
                "person_emp_length"
            ).value
        ),

        loan_intent:
            document.getElementById(
                "loan_intent"
            ).value,

        loan_grade:
            document.getElementById(
                "loan_grade"
            ).value,

        loan_amnt: Number(
            document.getElementById(
                "loan_amnt"
            ).value
        ),

        loan_int_rate: Number(
            document.getElementById(
                "loan_int_rate"
            ).value
        ),

        loan_percent_income: Number(
            document.getElementById(
                "loan_percent_income"
            ).value
        ),

        cb_person_default_on_file:
            document.getElementById(
                "cb_person_default_on_file"
            ).value,

        cb_person_cred_hist_length: Number(
            document.getElementById(
                "cb_person_cred_hist_length"
            ).value
        )
    };


    try {

        // ====================================================
        // Send Prediction Request to FastAPI
        // ====================================================

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)
        });


        // ====================================================
        // Check API Response
        // ====================================================

        if (!response.ok) {
            throw new Error(
                "Prediction request failed"
            );
        }


        const result = await response.json();


        // ====================================================
        // Stop Loading State
        // ====================================================

        loadingMessage.classList.add("hidden");

        submitButton.disabled = false;

        submitButton.textContent =
            "Predict Credit Risk";


        // ====================================================
        // Show Prediction Result
        // ====================================================

        resultBox.classList.remove("hidden");


        // ====================================================
        // Prediction Status
        // ====================================================

        const predictionText =
            document.getElementById(
                "predictionText"
            );

        const predictionDescription =
            document.getElementById(
                "predictionDescription"
            );

        const predictionBox =
            document.querySelector(
                ".prediction-box"
            );


        // Remove previous risk classes

        predictionBox.classList.remove(
            "low-risk",
            "high-risk"
        );


        if (result.prediction === 1) {

            predictionText.textContent =
                "High Default Risk";

            predictionDescription.textContent =
                "The model predicts a higher probability of loan default.";

            predictionBox.classList.add(
                "high-risk"
            );

        } else {

            predictionText.textContent =
                "Low Default Risk";

            predictionDescription.textContent =
                "The model predicts a lower probability of loan default.";

            predictionBox.classList.add(
                "low-risk"
            );
        }


        // ====================================================
        // Display Probabilities
        // ====================================================

        document.getElementById(
            "nonDefaultProbability"
        ).textContent =
            `${(
                result.non_default_probability * 100
            ).toFixed(2)}%`;


        document.getElementById(
            "defaultProbability"
        ).textContent =
            `${(
                result.default_probability * 100
            ).toFixed(2)}%`;


        // ====================================================
        // Update Probability Bars
        // ====================================================

        const nonDefaultPercentage =
            result.non_default_probability * 100;

        const defaultPercentage =
            result.default_probability * 100;


        document.getElementById(
            "nonDefaultBar"
        ).style.width =
            `${nonDefaultPercentage}%`;


        document.getElementById(
            "defaultBar"
        ).style.width =
            `${defaultPercentage}%`;


        // ====================================================
        // Explanation Container
        // ====================================================

        const explanationsContainer =
            document.getElementById(
                "explanations"
            );

        explanationsContainer.innerHTML = "";


        // ====================================================
        // Customer-Friendly Feature Names
        // ====================================================

        const featureNames = {

            person_age:
                "Age",

            person_income:
                "Annual Income",

            person_home_ownership:
                "Home Ownership",

            person_emp_length:
                "Employment Length",

            loan_intent:
                "Loan Purpose",

            loan_grade:
                "Loan Grade",

            loan_amnt:
                "Loan Amount",

            loan_int_rate:
                "Interest Rate",

            loan_percent_income:
                "Loan-to-Income Ratio",

            cb_person_default_on_file:
                "Previous Default History",

            cb_person_cred_hist_length:
                "Credit History Length"
        };


        // ====================================================
        // Find Highest SHAP Importance
        // ====================================================

        const maxImportance =
            Math.max(
                ...result.explanations.map(
                    item => item.importance
                )
            );


        // ====================================================
        // Format Customer Feature Values
        // ====================================================

        function formatFeatureValue(
            feature,
            value
        ) {

            const numericValue =
                Number(value);


            if (
                feature === "person_income" &&
                !isNaN(numericValue)
            ) {

                return numericValue.toLocaleString(
                    "en-US"
                );
            }


            if (
                feature === "loan_amnt" &&
                !isNaN(numericValue)
            ) {

                return numericValue.toLocaleString(
                    "en-US"
                );
            }


            if (
                feature ===
                    "cb_person_cred_hist_length" &&
                !isNaN(numericValue)
            ) {

                return `${numericValue} years`;
            }


            if (
                feature === "person_emp_length" &&
                !isNaN(numericValue)
            ) {

                return `${numericValue} years`;
            }


            if (
                feature === "loan_int_rate" &&
                !isNaN(numericValue)
            ) {

                return `${numericValue}%`;
            }


            if (
                feature ===
                    "loan_percent_income" &&
                !isNaN(numericValue)
            ) {

                return numericValue.toFixed(2);
            }


            return value;
        }


        // ====================================================
        // Create SHAP Explanation Cards
        // ====================================================

        result.explanations.forEach(
            (item) => {

                const explanation =
                    document.createElement(
                        "div"
                    );


                explanation.className =
                    "explanation-item";


                // Friendly Feature Name

                const displayName =
                    featureNames[item.feature] ||
                    item.feature;


                // Formatted Feature Value

                const displayValue =
                    formatFeatureValue(
                        item.feature,
                        item.value
                    );


                // =================================================
                // Impact Text
                // =================================================

                const impactText =
                    item.impact ===
                        "increased_default_risk"

                        ? "May increase default risk"

                        : "May reduce default risk";


                // =================================================
                // Impact CSS Class
                // =================================================

                const impactClass =
                    item.impact ===
                        "increased_default_risk"

                        ? "risk-increased"

                        : "risk-decreased";


                // =================================================
                // Impact Icon
                // =================================================

                const impactIcon =
                    item.impact ===
                        "increased_default_risk"

                        ? "↑"

                        : "↓";


                // =================================================
                // Relative SHAP Importance
                // =================================================

                const percentage =
                    maxImportance > 0

                        ? (
                            item.importance /
                            maxImportance
                        ) * 100

                        : 0;


                // =================================================
                // Explanation Card HTML
                // =================================================

                explanation.innerHTML = `

                    <div class="explanation-header">

                        <strong>
                            ${displayName}
                        </strong>

                        <span>
                            ${displayValue}
                        </span>

                    </div>


                    <div class="${impactClass}">

                        ${impactIcon}
                        ${impactText}

                    </div>


                    <div class="importance-label">

                        Model impact

                    </div>


                    <div class="importance-bar">

                        <div
                            class="importance-fill ${impactClass}"
                            style="width: ${percentage}%"
                        ></div>

                    </div>

                `;


                explanationsContainer.appendChild(
                    explanation
                );
            }
        );


    } catch (error) {

        // ====================================================
        // Handle Prediction Error
        // ====================================================

        console.error(
            "Prediction Error:",
            error
        );


        loadingMessage.classList.add(
            "hidden"
        );


        errorMessage.classList.remove(
            "hidden"
        );


        submitButton.disabled = false;

        submitButton.textContent =
            "Predict Credit Risk";
    }

});


// ============================================================
// Reset Form for a New Prediction
// ============================================================

newPredictionButton.addEventListener(
    "click",
    function () {

        // ====================================================
        // Clear Customer Inputs
        // ====================================================

        form.reset();


        // ====================================================
        // Hide Previous Prediction Result
        // ====================================================

        resultBox.classList.add(
            "hidden"
        );


        // ====================================================
        // Hide Error Message
        // ====================================================

        errorMessage.classList.add(
            "hidden"
        );


        // ====================================================
        // Hide Loading Message
        // ====================================================

        loadingMessage.classList.add(
            "hidden"
        );


        // ====================================================
        // Reset Prediction Button
        // ====================================================

        submitButton.disabled = false;

        submitButton.textContent =
            "Predict Credit Risk";


        // ====================================================
        // Scroll to Customer Information
        // ====================================================

        const customerInformation =
            document.getElementById(
                "customerInformation"
            );


        if (customerInformation) {

            customerInformation.scrollIntoView({

                behavior: "smooth",

                block: "start"
            });

        }

    }
);