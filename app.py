
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------
st.set_page_config(
    page_title="Appointment Cancellation Prediction",
    page_icon="💇",
    layout="wide"
)

# --------------------------------
# LOAD MODEL AND DATASET
# --------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_cancellation_model.pkl")

@st.cache_data
def load_dataset():
    possible_paths = [
        "dataset/chennai_salon_appointments_synthetic.csv",
        "C:/Users/Admin/Desktop/Salon AppointmentCancellation Prediction(ML)/chennai_salon_appointments_synthetic(dataset).csv"
    ]

    for path in possible_paths:
        if Path(path).exists():
            data = pd.read_csv(path)
            data.columns = data.columns.str.strip().str.lower()
            return data

    return None

try:
    model = load_model()
    df = load_dataset()

    if df is None:
        st.error("Dataset CSV file not found. Please check the dataset folder.")
        st.stop()

except Exception as e:
    st.error(f"Loading Error: {e}")
    st.stop()


# --------------------------------
# HELPER FUNCTION FOR DROPDOWNS
# --------------------------------
def get_options(column):
    if column in df.columns:
        values = df[column].dropna().astype(str).unique().tolist()
        return sorted(values)

    return ["Unknown"]


# --------------------------------
# MAIN PAGE
# --------------------------------
st.title("💇 Appointment Cancellation Prediction")

st.write(
    "Enter customer appointment details "
    "to predict whether the appointment will be cancelled or completed."
)

st.divider()

st.subheader("📝 Customer Appointment Details")

with st.form("appointment_form"):

    col1, col2 = st.columns(2)

    with col1:
        booking_date = st.date_input("Booking Date")

        appointment_date = st.date_input("Appointment Date")

        area = st.selectbox(
            "Select Area",
            get_options("area")
        )

        city = st.selectbox(
            "Select City",
            get_options("city")
        )

        salon_name = st.selectbox(
            "Select Salon Name",
            get_options("salon_name")
        )

    with col2:
        service = st.selectbox(
            "Select Service",
            get_options("service")
        )

        staff_member = st.selectbox(
            "Select Staff Member",
            get_options("staff_member")
        )

        booking_channel = st.selectbox(
            "Select Booking Channel",
            get_options("booking_channel")
        )

        service_price = st.number_input(
            "Service Price (₹)",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

    submit = st.form_submit_button(
        "🔍 Predict Appointment",
        use_container_width=True
    )


# --------------------------------
# PREDICTION
# --------------------------------
if submit:

    if appointment_date < booking_date:
        st.error("Appointment Date cannot be before Booking Date.")

    else:
        days_in_advance = (
            appointment_date - booking_date
        ).days

        # Prepare input values
        input_data = {
            "booking_date": str(booking_date),
            "appointment_date": str(appointment_date),
            "area": area,
            "city": city,
            "salon_name": salon_name,
            "service": service,
            "staff_member": staff_member,
            "booking_channel": booking_channel,
            "service_price_inr": service_price,
            "days_in_advance": days_in_advance,
            "cancelled_by": "Unknown"
        }

        try:
            expected_columns = (
                model.named_steps["preprocessor"].feature_names_in_
            )

            # Fill any additional training features
            for column in expected_columns:
                if column not in input_data:

                    if column in df.columns:
                        if pd.api.types.is_numeric_dtype(df[column]):
                            input_data[column] = df[column].median()
                        else:
                            mode_values = df[column].mode()
                            input_data[column] = (
                                mode_values.iloc[0]
                                if not mode_values.empty
                                else "Unknown"
                            )
                    else:
                        input_data[column] = 0

            new_booking = pd.DataFrame([input_data])

            # Keep exact training column order
            new_booking = new_booking.reindex(
                columns=expected_columns
            )

            # Predict
            prediction = model.predict(new_booking)[0]

            probability = model.predict_proba(
                new_booking
            )[0][1]

            # Save result
            st.session_state["prediction"] = int(prediction)
            st.session_state["probability"] = float(probability)
            st.session_state["days"] = days_in_advance

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# --------------------------------
# RESULT SECTION
# --------------------------------
if "prediction" in st.session_state:

    st.divider()

    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    prediction = st.session_state["prediction"]
    probability = st.session_state["probability"]

    with result_col1:

        if prediction == 1:
            st.error("❌ Appointment Cancelled")
        else:
            st.success("✅ Appointment Completed")

    with result_col2:
        st.metric(
            "Cancellation Probability",
            f"{probability * 100:.2f}%"
        )

    st.progress(probability)

    st.info(
        f"Appointment booked {st.session_state['days']} days in advance."
    )

    if st.button("Clear Prediction"):
        del st.session_state["prediction"]
        del st.session_state["probability"]
        del st.session_state["days"]
        st.rerun()


